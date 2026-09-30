from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import JsonResponse

from .models import ChatMessage

import sys
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(settings.BASE_DIR).parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT EMOTION ENGINE
# =========================================================

from src.emotion_engine import EmotionEngine


# =========================================================
# IMPORT CHATBOT ENGINE
# =========================================================

from chatbot.chatbot_engine import ChatbotEngine


# =========================================================
# IMPORT LLM ENGINE
# =========================================================

from chatbot.llm_engine import LLMEngine


# =========================================================
# MODEL PATHS
# =========================================================

FACE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "face_landmarker.task"
)

EMOTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "emotion_cnn_improved.keras"
)


# =========================================================
# CREATE EMOTION ENGINE
# =========================================================

emotion_engine = EmotionEngine(
    str(FACE_MODEL_PATH),
    str(EMOTION_MODEL_PATH)
)


# =========================================================
# CREATE RULE-BASED CHATBOT ENGINE
# =========================================================

chatbot_engine = ChatbotEngine()


# =========================================================
# CREATE GEMINI LLM ENGINE
# =========================================================

llm_engine = LLMEngine()


# =========================================================
# HOME
# =========================================================

def home(request):

    return render(
        request,
        "home.html"
    )


# =========================================================
# REGISTER
# =========================================================

def register(request):

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        email = request.POST.get(
            "email"
        )

        password = request.POST.get(
            "password"
        )

        confirm_password = request.POST.get(
            "confirm_password"
        )

        # -------------------------------------------------
        # CHECK PASSWORD
        # -------------------------------------------------

        if password != confirm_password:

            messages.error(
                request,
                "Passwords do not match."
            )

            return redirect("register")

        # -------------------------------------------------
        # CHECK USERNAME
        # -------------------------------------------------

        if User.objects.filter(
            username=username
        ).exists():

            messages.error(
                request,
                "Username already exists."
            )

            return redirect("register")

        # -------------------------------------------------
        # CREATE USER
        # -------------------------------------------------

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        user.save()

        messages.success(
            request,
            "Account created successfully!"
        )

        return redirect("login")

    return render(
        request,
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

def user_login(request):

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        password = request.POST.get(
            "password"
        )

        # -------------------------------------------------
        # AUTHENTICATE USER
        # -------------------------------------------------

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            return redirect("home")

        else:

            messages.error(
                request,
                "Invalid username or password."
            )

            return redirect("login")

    return render(
        request,
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

def user_logout(request):

    logout(request)

    return redirect("home")


# =========================================================
# DASHBOARD
# =========================================================

@login_required
def dashboard(request):

    from collections import Counter

    # ---------------------------------------------------------
    # GET CURRENT USER'S CHAT MESSAGES
    # ---------------------------------------------------------

    messages = ChatMessage.objects.filter(
        user=request.user
    ).order_by("-created_at")

    # ---------------------------------------------------------
    # TOTAL CONVERSATIONS
    # ---------------------------------------------------------

    total_messages = messages.count()

    # ---------------------------------------------------------
    # MOST DETECTED EMOTION
    # ---------------------------------------------------------

    emotions = [
        message.detected_emotion
        for message in messages
        if message.detected_emotion
    ]

    if emotions:

        emotion_counter = Counter(
            emotions
        )

        most_emotion, emotion_count = (
            emotion_counter.most_common(1)[0]
        )

    else:

        most_emotion = "No Data"
        emotion_count = 0

    # ---------------------------------------------------------
    # MOST USED INTENT
    # ---------------------------------------------------------

    intents = [
        message.detected_intent
        for message in messages
        if message.detected_intent
    ]

    if intents:

        intent_counter = Counter(
            intents
        )

        most_intent, intent_count = (
            intent_counter.most_common(1)[0]
        )

    else:

        most_intent = "No Data"
        intent_count = 0

    # ---------------------------------------------------------
    # AVERAGE EMOTION CONFIDENCE
    # ---------------------------------------------------------

    confidences = [
        message.confidence
        for message in messages
        if message.confidence is not None
    ]

    if confidences:

        average_confidence = (
            sum(confidences)
            / len(confidences)
        ) * 100

    else:

        average_confidence = 0

    # ---------------------------------------------------------
    # RECENT CONVERSATIONS
    # ---------------------------------------------------------

    recent_messages = messages[:5]
    llm_count = messages.filter(
        response_source="llm"
    ).count()

    fallback_count = messages.filter(
        response_source="fallback"
    ).count()

    # ---------------------------------------------------------
    # DASHBOARD CONTEXT
    # ---------------------------------------------------------

    context = {

        "total_messages":
            total_messages,

        "most_emotion":
            most_emotion,

        "emotion_count":
            emotion_count,

        "most_intent":
            most_intent,

        "intent_count":
            intent_count,

        "average_confidence":
            average_confidence,

        "recent_messages":
            recent_messages,
        "llm_count": llm_count,
        "fallback_count": fallback_count,
    }

    return render(
        request,
        "dashboard.html",
        context
    )


# =========================================================
# EMOTION DETECTION PAGE
# =========================================================

@login_required
def emotion_detection(request):

    return render(
        request,
        "emotion_detection.html"
    )


# =========================================================
# EMOTION DETECTION API
# =========================================================

@login_required
def detect_emotion(request):

    # =====================================================
    # ONLY POST REQUEST ALLOWED
    # =====================================================

    if request.method != "POST":

        return JsonResponse(
            {
                "success": False,
                "error":
                    "POST request required."
            },
            status=405
        )

    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if "image" not in request.FILES:

        return JsonResponse(
            {
                "success": False,
                "error":
                    "No image received."
            },
            status=400
        )

    # =====================================================
    # GET IMAGE
    # =====================================================

    image_file = request.FILES["image"]

    image_bytes = image_file.read()

    # =====================================================
    # CONVERT IMAGE TO NUMPY ARRAY
    # =====================================================

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    # =====================================================
    # DECODE IMAGE USING OPENCV
    # =====================================================

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # =====================================================
    # CHECK IMAGE DECODING
    # =====================================================

    if frame is None:

        return JsonResponse(
            {
                "success": False,
                "error":
                    "Could not decode image."
            },
            status=400
        )

    # =====================================================
    # PROCESS EMOTION
    # =====================================================

    try:

        result = emotion_engine.process(
            frame
        )

    except Exception as e:

        print(
            "Emotion Engine Error:",
            e
        )

        return JsonResponse(
            {
                "success": False,
                "error":
                    "Emotion detection failed."
            },
            status=500
        )

    # =====================================================
    # NO FACE DETECTED
    # =====================================================

    if result is None:

        # Clear previous emotion

        request.session[
            "current_emotion"
        ] = None

        request.session[
            "current_confidence"
        ] = 0

        return JsonResponse(
            {
                "success": True,
                "face_detected": False,
                "emotion": None,
                "confidence": 0
            }
        )

    # =====================================================
    # GET CURRENT EMOTION
    # =====================================================

    current_emotion = result[
        "emotion"
    ]

    current_confidence = result[
        "confidence"
    ]

    # =====================================================
    # STORE EMOTION IN SESSION
    # =====================================================

    request.session[
        "current_emotion"
    ] = current_emotion

    request.session[
        "current_confidence"
    ] = round(
        current_confidence,
        4
    )

    # =====================================================
    # RETURN EMOTION RESPONSE
    # =====================================================

    return JsonResponse(
        {
            "success": True,

            "face_detected":
                True,

            "emotion":
                current_emotion,

            "confidence":
                round(
                    current_confidence,
                    4
                )
        }
    )


# =========================================================
# CHAT PAGE
# =========================================================

@login_required
def chat_page(request):

    return render(
        request,
        "chat.html"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

@login_required
def chat_history(request):

    history = ChatMessage.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "chat_history.html",
        {
            "history": history
        }
    )


# =========================================================
# CHAT API
# =========================================================

# =========================================================
# CHAT API
# =========================================================

# =========================================================
# CHAT API
# =========================================================

@login_required
def chat_api(request):

    # ---------------------------------------------------------
    # ONLY POST REQUEST ALLOWED
    # ---------------------------------------------------------

    if request.method != "POST":

        return JsonResponse(
            {
                "success": False,
                "error": "POST request required."
            },
            status=405
        )

    # ---------------------------------------------------------
    # GET USER MESSAGE
    # ---------------------------------------------------------

    message = request.POST.get(
        "message",
        ""
    ).strip()

    if not message:

        return JsonResponse(
            {
                "success": False,
                "error": "Message cannot be empty."
            },
            status=400
        )

    # ---------------------------------------------------------
    # GET CURRENT EMOTION
    # ---------------------------------------------------------

    emotion = request.POST.get(
        "emotion",
        "Neutral"
    ).strip()

    # ---------------------------------------------------------
    # GET EMOTION CONFIDENCE
    # ---------------------------------------------------------

    confidence = request.POST.get(
        "confidence",
        "0"
    ).strip()

    try:

        confidence_value = float(
            confidence
        )

    except (ValueError, TypeError):

        confidence_value = 0.0

    # ---------------------------------------------------------
    # STEP 1: DETECT INTENT
    # Existing rule-based chatbot
    # ---------------------------------------------------------

    intent = chatbot_engine.detect_intent(
        message
    )

    # ---------------------------------------------------------
    # STEP 2: GET PREVIOUS CONVERSATION
    # ---------------------------------------------------------

    previous_messages = ChatMessage.objects.filter(
        user=request.user
    ).order_by("-created_at")[:10]

    previous_messages = list(
        reversed(previous_messages)
    )

    history = []

    for item in previous_messages:

        history.append(
            {
                "user": item.user_message,
                "assistant": item.bot_response
            }
        )

    # ---------------------------------------------------------
    # STEP 3: TRY GEMINI LLM
    # ---------------------------------------------------------

    try:

        response = llm_engine.generate_response(

            user_message=message,

            emotion=emotion,

            confidence=confidence_value,

            history=history

        )

        response_source = "llm"

    # ---------------------------------------------------------
    # STEP 4: FALLBACK CHATBOT
    # ---------------------------------------------------------

    except Exception as e:

        print(
            "Gemini Error:",
            e
        )

        intent, response = chatbot_engine.chat(

            message,

            emotion

        )

        response_source = "fallback"

    # ---------------------------------------------------------
    # STEP 5: SAVE CONVERSATION
    # ---------------------------------------------------------

    ChatMessage.objects.create(

        user=request.user,

        user_message=message,

        detected_emotion=emotion,

        confidence=confidence_value,

        detected_intent=intent,

        bot_response=response,

        response_source=response_source

    )

    # ---------------------------------------------------------
    # STEP 6: RETURN RESPONSE TO FRONTEND
    # ---------------------------------------------------------

    return JsonResponse(
        {
            "success": True,

            "intent": intent,

            "emotion": emotion,

            "confidence": confidence_value,

            "response": response,

            "source": response_source
        }
    )