import urllib.request
import urllib.parse
import json
import uuid
import sys

file_path = 'static/upload/10112_left.jpeg'
boundary = uuid.uuid4().hex
headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}

with open(file_path, 'rb') as f:
    file_data = f.read()

data = (
    b'--' + boundary.encode() + b'\r\n'
    b'Content-Disposition: form-data; name="file"; filename="test.jpeg"\r\n'
    b'Content-Type: image/jpeg\r\n\r\n' +
    file_data +
    b'\r\n--' + boundary.encode() + b'--\r\n'
)

try:
    req = urllib.request.Request('http://127.0.0.1:8080/predict', data=data, headers=headers)
    res = urllib.request.urlopen(req)
    print("Status:", res.getcode())
    response_data = json.loads(res.read())
    print("Keys:", response_data.keys())
    print("gradcam_overlay:", response_data.get('gradcam_overlay'))
    print("best_layer:", response_data.get('gradcam_comparison', {}).get('best_layer'))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
