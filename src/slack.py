from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
import os
import requests

from dotenv import load_dotenv
from config import SLACK_CHANNEL

load_dotenv()

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


class SlackBot:
    def __init__(self):
        self.app = App(token=SLACK_BOT_TOKEN)
        self.client = WebClient(token=SLACK_BOT_TOKEN)
        self.app.event("app_mention")(self.handle_mention)
        self.channel = "C09M5FPTSLV"
        

    def send_message(self, message):
        try:
            resp = self.app.client.chat_postMessage(channel=self.channel, text=message)
            return resp
        except Exception as e:
            raise

    def send_video(self, filepath, filename):
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return
        try:
            filesize = os.path.getsize(filepath)
            upload_resp = self.client.files_getUploadURLExternal(filename=filename, length=filesize)
            upload_url = upload_resp["upload_url"]
            file_id = upload_resp["file_id"]

            with open(filepath, "rb") as f:
                r = requests.post(upload_url, files={"file": f})
                r.raise_for_status()
            files = [{"id":file_id, "title":filename}]
            upload_resp = self.client.files_completeUploadExternal(files=files, channels=[self.channel])
            print(f"Video sent: {upload_resp}")
        except Exception as e:
            print(f"Failed to send video: {e}")

    def handle_mention(self, body, say):
        say("Death to squirrels!")

    def start(self, ):
        self.send_message("Bot started.")
        self.handler = SocketModeHandler(self.app, SLACK_APP_TOKEN)
        self.handler.start()

    

if __name__ == "__main__":
    slack_bot = SlackBot()
    slack_bot.start()
