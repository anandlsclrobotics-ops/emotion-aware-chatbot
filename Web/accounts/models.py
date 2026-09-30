from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class ChatMessage(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    user_message = models.TextField()

    detected_emotion = models.CharField(
        max_length=20,
        null=True,
        blank=True
    )

    confidence = models.FloatField(
        null=True,
        blank=True
    )

    detected_intent = models.CharField(
        max_length=50,
        null=True,
        blank=True
    )

    bot_response = models.TextField()

    response_source = models.CharField(
        max_length=20,
        default="fallback"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.user.username} - {self.detected_intent}"
