import os
import sys
import json
from flask import Flask, request
from flaskext.uploads import *
sys.path.append('../lib')
from improc import VisualObjectMatcher

app = Flask(__name__)
app.config['UPLOADED_IMAGES_DEST'] = "static/images/uploads"
images = UploadSet('images', IMAGES)
configure_uploads(app, (images,))

@app.route("/")
def root():
    return app.send_static_file("index.html")

@app.route("/upload",methods=['GET','POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        print("this is the grayscale value")
        print(request.form.get("grayscale"))
        print("this is the texture value")
        print(request.form.get("texture"))
        filename = images.save(request.files['photo'])
        proj_path = os.path.dirname(os.path.realpath(__file__)).replace("flask","")
        prd_args = {'base_folder': proj_path,'query_image_padding':50,'query_image_folder':'uploads'}
        prd = VisualObjectMatcher(prd_args, False)
        res = prd.match(proj_path + 'flask/static/images/uploads/' + filename)
        f = open('workfile', 'w')
        f.write(str(res))
        f.close()
        return app.send_static_file("upload.html")
    else:
        if os.path.isfile('workfile'):
            os.remove('workfile')
        return app.send_static_file("upload.html")

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=1234)