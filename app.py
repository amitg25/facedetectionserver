import queue
from os import fspath

from flask import Flask, request, jsonify, safe_join, current_app
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_uploads import IMAGES
from flask_uploads import UploadSet
from flask import make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from queue import Queue

import turicreate as tc
import sys
import os
import uuid
import logging
from flask import send_file
import threading
from marshmallow import fields
from marshmallow import post_load
from werkzeug.exceptions import NotFound, BadRequest

app = Flask(__name__)
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    #app.run(host='192.168.1.20', debug=True)

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] - %(threadName)-10s : %(message)s')

app.config['UPLOADED_IMAGES_DEST'] = './images'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

app.config['MODEL_DEST'] = './models'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'facerecognition.sqlite')
db = SQLAlchemy(app)
ma = Marshmallow(app)

users_models = db.Table('users_models', db.Column("user_id", db.Integer, db.ForeignKey('user.id')),
                        db.Column("model_id", db.Integer, db.ForeignKey('model.version')))


class Model(db.Model):
    version = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(100))
    users = db.relationship('User', secondary=users_models)


# User and Model Schemas

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    position = db.Column(db.String(300))

    def __init__(self, name, position):
        self.name = name
        self.position = position


class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'position')


class ModelSchema(ma.Schema):
    version = fields.Int()
    url = fields.Method("add_host_to_url")
    users = ma.Nested(UserSchema, many=True)

    def add_host_to_url(selfself, obj):
        return request.host_url + obj.url


user_schema = UserSchema()
users_schema = UserSchema(many=True)
model_schema = ModelSchema()
models_schema = ModelSchema(many=True)
db.create_all()


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.route("/vmware/face-recognition/api/v1.0/user/register", methods=['POST'])
def register_user():
    if not request.form or not 'name' in request.form:
        return make_response(jsonify({'status': 'failed', 'error': 'bad request', 'message': 'Name is required'}), 400)
    else:
        name = request.form['name']
        position = request.form.get('position')
        if position is None:
            position = ""
        newuser = User(name, position)
        db.session.add(newuser)
        db.session.commit()
        if 'photos' in request.files:
            uploaded_images = request.files.getlist('photos')
            save_images_to_folder(uploaded_images, newuser)
        return jsonify({'status': 'success', 'user': user_schema.dump(newuser)})


@app.route("/vmware/face-recognition/api/v1.0/model/info", methods=['GET'])
def get_model_info():
    models_schema.context['request'] = request
    model = Model.query.order_by(Model.version.desc()).first()
    if model is None:
        return make_response(jsonify({'status': 'failed', 'error': 'model is not ready'}), 400)
    else:
        return jsonify({'status': 'success', 'model': model_schema.dump(model)})


@app.route("/models/<path:filename>")
def download(filename):
    #return send_model('models', filename, as_attachment=True)
    sendFile = "models/"+filename
    return send_file(sendFile, attachment_filename=filename)


def save_images_to_folder(images_to_save, user):
    for a_file in images_to_save:
        images.save(a_file, str(user.id), str(uuid.uuid4()) + '.')

    model = Model.query.order_by(Model.version.desc()).first()

    if model is not None:
        queue.put(model.version + 1)
    else:
        queue.put(1)


def send_model(directory, filename, **options):
    logging.debug("Sending Models.....")
    filename = os.fspath(filename)
    directory = fspath(directory)
    filename = safe_join(directory, filename)
    if not os.path.isabs(filename):
        filename = os.path.join(current_app.root_path, filename)
    try:
        if not os.path.isfile(filename):
            raise NotFound()
    except (TypeError, ValueError):
        raise BadRequest()

    logging.debug("Final Path-----" + filename)
    options.setdefault("conditional", True)

#TuriCreate (iOS)
#tensorflow (Android)
def train_model():
    while True:
        version = queue.get()
        logging.debug('loading images')
        data = tc.image_analysis.load_images('images', with_path=True)

        data['label'] = data['path'].apply(lambda path: path.split('/')[-2])

        filename = 'Faces_v' + str(version)
        mlmodel_filename = filename + '.mlmodel'
        models_folder = 'models/'

        data.save(models_folder + filename + '.sframe')

        result_data = tc.SFrame(models_folder + filename + '.sframe')
        # train_data = result_data.random_split(0.8)

        # model = tc.image_classifier.create(train_data, target='label', model='resnet-50', max_iterations=100, verbose=True)
        model = tc.image_classifier.create(result_data, target='label', model='resnet-50', max_iterations=100,
                                           verbose=True)
        db.session.commit()
        logging.debug('saving model')

        model.save(models_folder + filename + '.model')
        logging.debug('saving coremlmodel')
        model.export_coreml(models_folder + mlmodel_filename)

        modelData = Model()
        modelData.url = models_folder + mlmodel_filename
        classes = model.classes
        for userId in classes:
            user = User.query.get(userId)
            if user is not None:
                modelData.users.append(user)
        db.session.add(modelData)
        db.session.commit()
        logging.debug('done creating model')
        queue.task_done()


queue = Queue(maxsize=0)
thread = threading.Thread(target=train_model, name='TrainingDaemon')
thread.setDaemon(False)
thread.start()
