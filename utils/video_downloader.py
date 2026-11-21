import shutil

import yt_dlp

URL = "https://www.youtube.com/watch?v=r6iz35hizA8"
OUTPUT_TEMPLATE = "downloads/%(title)s [%(id)s].%(ext)s"
MAX_HEIGHT = 720
BROWSER = "firefox"
VERBOSE = False
DENO_PATH = shutil.which("deno") or r"C:/Users/robin/.deno/bin/deno.exe"

FORMAT = (
    f"bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]"
    f"/bv*[height<={MAX_HEIGHT}]+ba"
    f"/b[height<={MAX_HEIGHT}]"
)

BASE_OPTS = {
    "format": FORMAT,
    "merge_output_format": "mp4",
    "outtmpl": OUTPUT_TEMPLATE,
    "continuedl": True,
    "retries": 10,
    "fragment_retries": 10,
    "quiet": False,
    "verbose": VERBOSE,
    "noprogress": False,
    "js_runtimes": {"deno": {"path": DENO_PATH}},
}


def download_video(url: str = URL, options: dict | None = None) -> None:
    ydl_opts = options or BASE_OPTS
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    download_video()
