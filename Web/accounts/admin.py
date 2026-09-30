from django.contrib import admin
from .models import ChatMessage


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):

    # Columns shown in admin list
    list_display = (
        "user",
        "user_message",
        "detected_emotion",
        "confidence",
        "detected_intent",
        "created_at",
    )

    # Filters on right side
    list_filter = (
        "detected_emotion",
        "detected_intent",
        "created_at",
    )

    # Search box
    search_fields = (
        "user__username",
        "user_message",
        "bot_response",
        "detected_intent",
    )

    # Latest chats first
    ordering = (
        "-created_at",
    )

    # Fields shown when opening one chat
    readonly_fields = (
        "created_at",
    )