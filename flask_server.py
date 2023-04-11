import os
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import prediction
import mlflow.tensorflow

# Set up Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tf.keras.backend.clear_session()
# If model_uri is not provided, get the most recent run ID for the 'unet' experiment
client = mlflow.tracking. MlflowClient()
exp = client.get_experiment_by_name("unet")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
runs.sort_values('metrics.val_dice_coef', ascending=False, inplace=True)
run_id = runs['run_id'].iloc[0]

# load model from MLflow
model_path = os.path.join("runs:/", run_id, "model")
model = mlflow.tensorflow.load_model(model_path)

# Define function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define Flask route
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file selected')
        file = request.files['file']
        # Check if file has allowed extension
        #if not allowed_file(file.filename):
        #    return render_template('index.html', message='Only nii, nii.gz files are allowed')
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Make prediction
        prediction.showPredicts(model, file_path)
        # Show predicted image on the same page
        return render_template('index.html', image_file="pred_plot.png")
    # Render initial page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
