class Logger:
    def __init__(self, slack_logging, slack_bot=None):
        self._slack_bot = slack_bot
        self._slack_logging = slack_logging
        # self._recording_dir = recording_dir

    def info(self, message):
        print(message)
        if self._slack_logging:
            self._slack_bot.send_message(message)

    def video(self, video_dir, video_filename):
        print(f"Storing video: {video_dir}{video_filename}")
        if self._slack_logging:
            self._slack_bot.send_video(video_dir, video_filename)
