from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
import os
import requests

from dotenv import load_dotenv
load_dotenv()

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


class SlackBot:
    def __init__(self, slack_channel):
        self._app = App(token=SLACK_BOT_TOKEN)
        self._client = WebClient(token=SLACK_BOT_TOKEN)
        self._app.event("app_mention")(self.handle_mention)
        self._channel = slack_channel

    def send_message(self, message):
        try:
            resp = self._app.client.chat_postMessage(channel=self._channel, text=message)
            return resp
        except Exception:
            raise

    def send_video(self, video_dir, video_filename):
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"File does not exist: {video_path}")
            return
        try:
            filesize = os.path.getsize(video_path)
            upload_resp = self._client.files_getUploadURLExternal(
                filename=video_filename, length=filesize
            )
            upload_url = upload_resp["upload_url"]
            file_id = upload_resp["file_id"]

            with open(video_path, "rb") as f:
                r = requests.post(upload_url, files={"file": f})
                r.raise_for_status()
            files = [{"id": file_id, "title": video_filename}]
            upload_resp = self._client.files_completeUploadExternal(
                files=files, channels=[self._channel]
            )
            print(f"Video sent: {upload_resp}")
        except Exception as e:
            print(f"Failed to send video: {e}")

    def handle_mention(self, body, say):
        say("Death to squirrels!")

    def start_handler(self):
        self.send_message("Bot started.")
        self._handler = SocketModeHandler(self._app, SLACK_APP_TOKEN)
        self._handler.start()


if __name__ == "__main__":
    slack_bot = SlackBot()
    slack_bot.start_handler()
