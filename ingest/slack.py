import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
from typing import List, Dict, Any

from config import SLACK_BOT_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackIngest:
    def __init__(self):
        self.client = WebClient(token=SLACK_BOT_TOKEN)
    
    def get_channels(self):
        try:
            result = self.client.conversations_list()
            return result["channels"]
        except SlackApiError as e:
            logger.error(f"Error getting channels: {e}")
            return []
    
    def get_messages(self, channel_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Extract messages from a Slack channel"""
        try:
            result = self.client.conversations_history(channel=channel_id, limit=limit)
            messages = result["messages"]
            
            # Process messages to extract useful information
            processed_messages = []
            for msg in messages:
                if "text" in msg and msg["text"]:
                    # Filter for bug reports - you can customize this logic
                    if "bug" in msg["text"].lower() or "issue" in msg["text"].lower() or "error" in msg["text"].lower():
                        processed_msg = {
                            "text": msg["text"],
                            "ts": msg["ts"],
                            "date": datetime.fromtimestamp(float(msg["ts"])).strftime('%Y-%m-%d %H:%M:%S'),
                            "user": self._get_user_info(msg.get("user", ""))
                        }
                        # Add thread replies if they exist
                        if "thread_ts" in msg:
                            replies = self._get_thread_replies(channel_id, msg["thread_ts"])
                            processed_msg["replies"] = replies
                        
                        processed_messages.append(processed_msg)
            
            return processed_messages
        
        except SlackApiError as e:
            logger.error(f"Error fetching messages: {e}")
            return []
    
    def _get_user_info(self, user_id: str) -> Dict[str, str]:
        """Get user information for a given user ID"""
        try:
            if not user_id:
                return {"name": "Unknown", "real_name": "Unknown User"}
            
            result = self.client.users_info(user=user_id)
            user = result["user"]
            return {
                "name": user.get("name", "Unknown"),
                "real_name": user.get("real_name", "Unknown User")
            }
        except SlackApiError:
            return {"name": "Unknown", "real_name": "Unknown User"}
    
    def _get_thread_replies(self, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Get replies to a thread"""
        try:
            result = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )
            # Skip the first message as it's the parent
            replies = result["messages"][1:] if len(result["messages"]) > 1 else []
            
            processed_replies = []
            for reply in replies:
                processed_reply = {
                    "text": reply["text"],
                    "ts": reply["ts"],
                    "date": datetime.fromtimestamp(float(reply["ts"])).strftime('%Y-%m-%d %H:%M:%S'),
                    "user": self._get_user_info(reply.get("user", ""))
                }
                processed_replies.append(processed_reply)
                
            return processed_replies
        
        except SlackApiError as e:
            logger.error(f"Error fetching thread replies: {e}")
            return [] 