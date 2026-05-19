[app]

title = AuraRAG

package.name = aurarag

package.domain = org.aurarag

source.dir = .

source.include_exts = py,png,jpg,kv,atlas,json,txt

version = 3.4

requirements = python3,streamlit,fastapi,uvicorn,langchain,chromadb,pandas

orientation = portrait

fullscreen = 0

android.api = 34

android.minapi = 24

android.sdk = 34

android.ndk = 25b

android.accept_sdk_license = True

android.permissions = INTERNET

android.archs = arm64-v8a, armeabi-v7a

android.allow_backup = True

android.enable_androidx = True

android.logcat_filters = *:S python:D

presplash.color = #111111

window.clearcolor = #111111

osx.python_version = 3

osx.kivy_version = 2.3.0