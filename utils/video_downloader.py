import yt_dlp
import shutil

URL = "https://www.youtube.com/watch?v=n2zjuuvgkGY"
OUTPUT_TEMPLATE = "downloads/%(title)s [%(id)s].%(ext)s"
MAX_HEIGHT = 720
 
BROWSER = "firefox"
 
VERBOSE = False
 
DENO_PATH = shutil.which("deno") or r"C:/Users/robin/.deno/bin/deno.exe"
 
# ------------------------------------------------------------------------
 
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

if __name__ == "__main__":
    with yt_dlp.YoutubeDL(BASE_OPTS) as ydl:
        ydl.download([URL])