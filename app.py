import db
import hashlib
import io
import os
import time
import yaml
import sys

from celery import Celery
import flask
from flask import Flask, request, make_response, jsonify, send_file
from werkzeug.utils import secure_filename

CONFIG_FILE = "config.yaml"


# Celery support with help from these links:
# https://flask.palletsprojects.com/en/2.0.x/patterns/celery/
# https://github.com/miguelgrinberg/flask-celery-example/blob/master/app.py

# Make celery object for communicating with task queue service
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['result_backend'],
        broker=app.config['CELERY_broker_url'],
        task_track_started=True
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)

# Read config file
try:
    with open(CONFIG_FILE) as fp:
        yaml_config = yaml.safe_load(fp)
        app.config.update(yaml_config)
except FileNotFoundError:
    print('\nERROR: config file "{}" not found! You need to create a config file.\n'
          'See the example in "config.yaml.example".\n'.format(CONFIG_FILE))
    sys.exit(1)

celery = make_celery(app)
db.init_app(app)


# Calculate sha1 hash for given file pointer
def calculate_hash(fp):
    hash = hashlib.sha1()
    while chunk := fp.read(8192):
        hash.update(chunk)
    result = hash.hexdigest()
    assert len(result) == 40, "Calculation of sha1 digest for file failed"
    return result


# Translate from short file path to real absolute path on server
def data_path(db_fname):
    return os.path.join(app.config['DATA_FOLDER'], db_fname)


# Handle file upload
def handle_upload(request):
    # Check if we are receiving a valid file upload (multipart/form-data)
    if 'file' not in request.files:
        return make_response(("No file given\n", 400))
    fp = request.files['file']
    if fp.filename == '':
        return make_response(("Empty file\n", 400))

    # Calculate SHA1 hash of file
    content_type = fp.content_type
    fname = secure_filename(fp.filename)
    hash_digest = calculate_hash(fp)

    # Open database
    con = db.get_db()
    cur = con.cursor()

    # Check if SHA1 already present in database
    file_exists = False
    cur.execute("SELECT filename FROM files WHERE file_id=?", (hash_digest,))
    found = cur.fetchone()
    if found:
        file_exists = os.path.exists(data_path(found['filename']))

    # If no SHA1 in db, or its corresponding file is missing...
    if not file_exists:
        subdir = hash_digest[:8]  # directory, inside DATA_FOLDER
        fname = os.path.join(subdir, fname)  # file inside DATA_FOLDER

        # Create dir if doesn't exist
        os.makedirs(os.path.dirname(data_path(fname)), exist_ok=True)

        # Save file
        fp.seek(0)
        fp.save(data_path(fname))

        cur.execute("REPLACE INTO files (filename, file_id, content_type) VALUES (?, ?, ?)",
                    (fname, hash_digest, content_type))
        con.commit()
    else:
        print('File {} with file_id {} already found in database'.format(
            fname, hash_digest))

    cur.close()
    return jsonify({'file_id': hash_digest}), 202


# Get file info from sqlite database
def get_fileinfo(file_id):
    con = db.get_db()
    cur = con.cursor()

    fields = ['file_id', 'filename', 'content_type', 'created']
    sql_query = "SELECT {} FROM files".format(','.join(fields))
    if file_id is None:  # if no file_id is given, we query ALL files
        cur.execute(sql_query)
    else:
        cur.execute(sql_query + " WHERE file_id LIKE ? ", (file_id + '%',))
    result = [{f: row[f] for f in fields}
              for row in cur.fetchall()]
    cur.close()
    return result


# Handle /file API endpoint
# POST: upload file
# GET: get list of all files on server
@app.route('/file', methods=['GET', 'POST'])
def get_all_files():
    if request.method == 'POST':
        return handle_upload(request)
    else:
        return jsonify(get_fileinfo(None)), 200


# Handle /file/<file_id> API endpoint
# GET: download single file
# DELETE: single file
@app.route('/file/<file_id>', methods=['GET', 'DELETE'])
def get_single_file(file_id):
    result = get_fileinfo(file_id)
    if len(result) == 0:
        return make_response(("No file with that id!\n", 400))
    if len(result) > 1:
        return make_response(("Ambiguous id, matched more than one file!\n", 400))

    fname = result[0]['filename']
    real_fname = data_path(fname)

    if request.method == 'DELETE':
        sha1 = result[0]['file_id']
        assert len(sha1) == 40  # sanity check

        con = db.get_db()
        cur = con.cursor()
        cur.execute("DELETE FROM files WHERE file_id = ?", (result[0]['file_id'],))
        con.commit()
        cur.close()

        try:
            os.remove(real_fname)
        except FileNotFoundError:
            print('WARNING: file to be removed not found: {}'.format(real_fname))

        return make_response(("Successfully deleted file {}\n".format(fname), 200))
    else:
        with open(real_fname, 'rb') as fp:
            print('Returning file', fname)
            bindata = fp.read()
            kwargs = { 'mimetype': result[0]['content_type']}
            if int(flask.__version__[0]) >= 2:  # because this was changed in Flask 2.0
                kwargs['download_name'] = os.path.basename(fname)
            else:
                kwargs['attachment_filename'] = os.path.basename(fname)
            return send_file(io.BytesIO(bindata), **kwargs)

    return make_response(("Unable to read data\n", 500))


# Add a file created by DeepLabCut to sqlite database (not uploaded from user)
def add_results_file(fname, content_type):
    # Calculate SHA1 from file
    hash_digest = calculate_hash(open(data_path(fname), 'rb'))

    # Open database
    con = db.get_db()
    cur = con.cursor()

    # Add database entry for CSV file
    cur.execute("REPLACE INTO files (filename, file_id, content_type) "
                "VALUES (?, ?, ?)", (fname, hash_digest, content_type))
    con.commit()
    cur.close()

    return hash_digest


# Actual task for analysing video with DeepLabCut
@celery.task
def analyse_video(fname, cfg_fname):
    import deeplabcut

    print('Starting analysis of video {} with model {}.'.format(fname, cfg_fname))
    res_id = deeplabcut.analyze_videos(cfg_fname,
                                       [data_path(fname)],
                                       save_as_csv=True)
    csv_fname = os.path.splitext(fname)[0] + res_id + '.csv'

    # CSV file may end with '_filtered.csv' if it has been created via triangulate
    if not os.path.exists(data_path(csv_fname)):
        csv_fname = os.path.splitext(fname)[0] + res_id + '_filtered.csv'

    assert os.path.exists(data_path(csv_fname)), "Unable to find CSV file with name " + csv_fname

    hash_digest = add_results_file(csv_fname, "text/csv")

    return {'csv_file': hash_digest}


@celery.task
def analyse_image(fname, cfg_fname):
    import deeplabcut
    
    print('Starting analysis of image {} with model {}.'.format(fname, cfg_fname))
    
    print(data_path(fname))
    
    res_id = deeplabcut.analyze_time_lapse_frames(cfg_fname,
                                       [data_path(fname)][0],
                                       frametype='.jpg',
                                       save_as_csv=True)
    print(res_id)
    
    csv_fname = os.path.splitext(fname)[0] + res_id + '.csv'
    print(csv_fname)
    assert os.path.exists(data_path(csv_fname)), "Unable to find CSV file with name " + csv_fname
    
    hash_digest = add_results_file(csv_fname, "text/csv")
    
    return {'csv_file': hash_digest}


# Task for labelling video with DeepLabCut
@celery.task
def create_labeled_video(fname, cfg_fname):
    import deeplabcut

    print('Creating labeled video for {} with model {}.'.format(fname, cfg_fname))

    from deeplabcut.utils import auxiliaryfunctions

    deeplabcut.create_labeled_video(cfg_fname,
                                    [data_path(fname)],
                                    draw_skeleton=True)

    cfg = auxiliaryfunctions.read_config(cfg_fname)
    res_id, _ = auxiliaryfunctions.GetScorerName(
        cfg, shuffle=1, trainFraction=cfg["TrainingFraction"][0])

    video_fname = os.path.splitext(fname)[0] + res_id + '_labeled.mp4'
    if not os.path.exists(data_path(video_fname)):
        video_fname = os.path.splitext(fname)[0] + res_id + 'filtered_labeled.mp4'

    assert os.path.exists(data_path(video_fname)), "Unable to find video file with name " +  video_fname

    hash_digest = add_results_file(video_fname, "video/mp4")

    return {'labeled_video': hash_digest}


# Task for triangulate 3D coordinates
@celery.task
def triangulate(fname1, fname2, cfg_3d_name):
    import deeplabcut
    
    print('Creating 3D coordinates for {} and {} with model {}.'.format(fname1, fname2, cfg_3d_name))

    with open(cfg_3d_name) as fp:
        cfg_3d = yaml.safe_load(fp)

    camera_names = cfg_3d['camera_names']
    scorername = cfg_3d['scorername_3d']  # DLC_3D

    assert len(camera_names) == 2, "Model should have 2 cameras, not " + str(len(camera_names))
    cn1 = camera_names[0]
    cn2 = camera_names[1]

    assert cn1 in fname1, "Video file name should contain '{}': {}".format(cn1, fname1)
    assert cn2 in fname2, "Video file name should contain '{}': {}".format(cn2, fname2)
    
    res_id = deeplabcut.triangulate(cfg_3d_name,
                                    [[data_path(fname1), data_path(fname2)]],
                                    save_as_csv=True)

    destfolder = os.path.dirname(data_path(fname1))
    vname = os.path.splitext(os.path.basename(fname2))[0]

    # The next lines are copied from deeplabcut.triangulate. Very
    # ugly, but the only way to get exactly the same output filename
    # as it doesn't return the filename...
    
    prefix = vname.split(cn2)[0]
    suffix = vname.split(cn2)[-1]
    if prefix == "":
        pass
    elif prefix[-1] == "_" or prefix[-1] == "-":
        prefix = prefix[:-1]
    
    if suffix == "":
        pass
    elif suffix[0] == "_" or suffix[0] == "-":
        suffix = suffix[1:]
    
    if prefix == "":
        output_file = os.path.join(destfolder, suffix)
    else:
        if suffix == "":
            output_file = os.path.join(destfolder, prefix)
        else:
            output_file = os.path.join(destfolder, prefix + "_" + suffix)
    
    csv_fname = output_file + "_" + scorername + ".csv"
    
    print(csv_fname)
    assert os.path.exists(data_path(csv_fname)), "Unable to find CSV file with name " + csv_fname
    
    hash_digest = add_results_file(csv_fname, "text/csv")
    
    return {'csv_file': hash_digest}


@celery.task
def label_image(fname, cfg_fname):
    import deeplabcut
    import cv2
    import os
    
    print('Creating labeled image {} with model {}.'.format(fname, cfg_fname))
    image_path = data_path(fname)
    folder_path = os.path.dirname(image_path)
    print(folder_path)
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith(".csv"):
            with open(folder_path + "/" + file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                for line in lines:
                    if i >= 3:
                        info = line.split(",")
                        #image name
                        image_name = fname.split("/")[1]
                        print(image_name)
                        #read detected points
                        piippu1_x = float(info[1])
                        piippu1_y = float(info[2])
                        piippu2_x = float(info[4])
                        piippu2_y = float(info[5])
                        haara_x =  float(info[7])
                        haara_y = float(info[8])
                        vasen_x = float(info[10])
                        vasen_y = float(info[11])
                        oikea_x = float(info[13])
                        oikea_y = float(info[14])

                        #read image
                        image = cv2.imread(image_path)
                    
                        #create points, color and thickness
                        piippu1 = (int(piippu1_x), int(piippu1_y))
                        piippu2 = (int(piippu2_x), int(piippu2_y))
                        haara = (int(haara_x), int(haara_y))
                        vasen = (int(vasen_x), int(vasen_y))
                        oikea = (int(oikea_x), int(oikea_y))
                        color = (0, 255, 0)
                        thickness = 4
                    
                        #draw lines to image
                        image = cv2.line(image, piippu1, piippu2, color, thickness)
                        image = cv2.line(image, piippu2, haara, color, thickness)
                        image = cv2.line(image, haara, vasen, color, thickness)
                        image = cv2.line(image, haara, oikea, color, thickness)
                        
                        image_name = folder_path + "/" + "labeled_" + image_name 
                        print(image_name)
                        #save image 
                        cv2.imwrite(image_name, image)
                    
                    i = i + 1
    
    assert os.path.exists(data_path(image_name)), "Unable to find image file with name: " + image_name
    hash_digest = add_results_file(image_name, "image/jpg")
    
    return {'labeled_image': hash_digest}


# Dummy task for testing
@celery.task
def analyse_sleep(fname, sleep_time):
    print('Sleeping for {} seconds...'.format(sleep_time))
    time.sleep(sleep_time)

    # Create CSV file
    import csv
    ldir = os.path.dirname(fname)
    csv_fname = os.path.join(ldir, 'output.csv')
    with open(data_path(csv_fname), 'w') as fp:
        w = csv.writer(fp)
        w.writerow(['sleep', sleep_time])

    # Calculate SHA1 from CSV file
    hash_digest = add_results_file(csv_fname, "text/csv")

    return {'sleep_time': sleep_time, 'csv_file': hash_digest}


# Get list of submitted analyses from sqlite database
def get_analysis_list():
    con = db.get_db()
    cur = con.cursor()

    fields = ['task_id', 'file_id', 'file2_id', 'analysis_name', 'state', 'created']
    cur.execute('SELECT id, {} FROM analyses'.format(', '.join(fields)))

    return [{f: row[f] for f in fields} for row in cur.fetchall()]


def get_file_from_db(file_id, cur):
    cur.execute('SELECT filename, file_id FROM files WHERE file_id LIKE ?', (file_id+'%',))
    found = cur.fetchone()
    return found['filename'] if found else None


# Handle /analysis API endpoint
# GET: get list of submitted analyses
# POST: add new analysis
@app.route('/analysis', methods=['GET', 'POST'])
def start_analysis():
    if request.method == 'GET':
        return jsonify(get_analysis_list()), 200

    # Check arguments
    if 'file_id' not in request.form:
        return make_response(("No file_id given\n", 400))
   
    file_id = request.form['file_id']
    if file_id == '':
    	return make_response(("No file_id given\n", 400))
    	
    analysis = request.form.get('analysis')

    dlc_model = request.form.get('model')

    if analysis is None:
        return make_response(("No analysis task given\n", 400))
        
    if analysis == '':
    	return make_response(("No analysis task given\n", 400))
    	
    if dlc_model == '':
    	return make_response(("No model task given\n", 400))
    analysis = analysis.lower()

    file2_id = None
    fname2 = None
    if analysis == 'triangulate':
        if 'file2_id' not in request.form:
            return make_response(("No file2_id given\n", 400))
        file2_id = request.form['file2_id']
        if file2_id == '':
    	    return make_response(("No file2_id given\n", 400))

    # Open database
    con = db.get_db()
    cur = con.cursor()

    # Find file_id in database
    fname = get_file_from_db(file_id, cur)
    if fname is None:
        return make_response(("Given file_id not found\n", 400))
    if not os.path.exists(data_path(fname)):
        return make_response(("File no longer exists in server\n", 400))

    if file2_id is not None:
        fname2 = get_file_from_db(file2_id, cur)
        if fname2 is None:
            return make_response(("Given file2_id not found\n", 400))
        if not os.path.exists(data_path(fname2)):
            return make_response(("File for file2_id no longer exists in server\n", 400))

    if analysis == 'sleep':
        sleep_time = request.form.get('time')
        if sleep_time is None:
            return make_response(("No time argument given\n", 400))
        task = analyse_sleep.apply_async(args=(fname, int(sleep_time)))
    elif analysis in ('video', 'label', 'image', 'label-image', 'triangulate'):
        if dlc_model is None:
            return make_response(("No model argument given. Supported models: {}.\n".format(
                ', '.join(app.config['DLC_MODELS'].keys())), 400))
        if dlc_model not in app.config['DLC_MODELS']:
            return make_response(("No model named '{}' found. Supported models: {}.\n".format(
                dlc_model, ', '.join(app.config['DLC_MODELS'].keys())), 400))
        dlc_model_path = app.config['DLC_MODELS'][dlc_model]
        if analysis == 'video':
            task = analyse_video.apply_async(args=(fname, dlc_model_path))
        elif analysis == 'image':
            task = analyse_video.apply_async(args=(fname, dlc_model_path))
        elif analysis == 'label':
            task = create_labeled_video.apply_async(args=(fname, dlc_model_path))
        elif analysis == 'label-image':
            task = label_image.apply_async(args=(fname, dlc_model_path))
        elif analysis == 'triangulate':
            task = triangulate.apply_async(args=(fname, fname2, dlc_model_path))
    else:
        return make_response(("Unknown analysis task '{}'\n".format(analysis)),
                             400)

    task_id = task.id
    cur.execute("INSERT INTO analyses (task_id, file_id, file2_id, analysis_name, state) "
                "VALUES (?, ?, ?, ?, ?)", (task_id, fname, fname2, analysis, "PENDING"))
    con.commit()

    return jsonify({'task_id': task_id}), 202


# Handle /analysis/<task_id> endpoint
# Get status from analysis in progress, returns results if analysis is done
@app.route('/analysis/<task_id>')
def get_analysis(task_id):
    con = db.get_db()
    cur = con.cursor()

    cur.execute('SELECT id, task_id, state, analysis_name FROM analyses WHERE task_id LIKE ?', (task_id+'%',))
    found = cur.fetchone()
    if not found:
        return make_response(("Given task_id not found\n", 400))

    db_id = found['id']
    task_id = found['task_id']
    analysis_name = found['analysis_name']

    task = analyse_video.AsyncResult(task_id)
    state = task.state

    if state != found['state']:
        cur.execute('UPDATE analyses SET state=? WHERE id=?', (state, db_id))
        con.commit()

    state_json = {'state': task.state}
    if task.state in ('PENDING', 'STARTED'):
        return jsonify(state_json), 202
    elif task.state == 'FAILURE':
        exception_str = repr(task.result)
        state_json['description'] = 'Python error: ' + exception_str
        if analysis_name == 'label' and exception_str.startswith('AxisError(2'):
            state_json['description'] = 'Bad video file format, please see: https://deeplabcut.github.io/DeepLabCut/docs/recipes/io.html#tips-on-video-re-encoding-and-preprocessing'
        elif type(task.result) is AssertionError:
            state_json['description'] = str(task.result)
        # state_json['traceback'] = task.traceback}
        return jsonify(state_json), 500
    elif task.state == 'SUCCESS':
        return jsonify(task.info)
    else:
        return make_response(("Unexpected task state: '{}'\n".format(task.state)), 500)


# Handle clean up, this can be run from command line with "flask clean-up"
# Deletes files older than config option clean_up_files_days
# Deletes analyses older than clean_up_analyses_days
@app.cli.command('clean-up')
def clean_up_command():
    clean_up_files_days = app.config['clean_up_files_days']
    clean_up_analyses_days = app.config['clean_up_analyses_days']

    print('Running clean up...')

    con = db.get_db()
    cur = con.cursor()

    cur.execute('SELECT * FROM analyses WHERE created < DATETIME("now", ?)',
                ('-{} days'.format(clean_up_analyses_days),))

    for row in cur.fetchall():
        task = analyse_video.AsyncResult(row['task_id'])
        if task.state in ('PENDING', 'STARTED'):
            print('WARNING: not deleting old task {} as status is {}'.format(
                row['task_id'], task.state))
        else:
            print('DELETING {} task {}, {}, {} ({})'.format(
                row['analysis_name'], row['task_id'], row['file_id'][:10], row['state'],
                row['created']))
            cur.execute('DELETE FROM analyses WHERE id = ?', (row['id'],))
            con.commit()
            task.forget()

    cur.execute('SELECT * FROM files WHERE created < DATETIME("now", ?)',
                ('-{} days'.format(clean_up_files_days),))

    for row in cur.fetchall():
        cur.execute('SELECT task_id FROM analyses WHERE file_id=?', (row['file_id'],))
        rows = cur.fetchall()
        if len(rows) > 0:
            print('WARNING: not deleting old file {} as it still has analyses:'.format(
                row['file_id']))
            for r in rows:
                print('  ', r[0])
        else:
            print('DELETING file {}, {} ({})'.format(row['file_id'], row['filename'],
                                                     row['created']))
            cur.execute('DELETE FROM files WHERE id = ?', (row['id'],))
            con.commit()

            real_fname = data_path(row['filename'])
            try:
                os.remove(real_fname)
            except FileNotFoundError:
                print('WARNING: file to be removed not found: {}'.format(real_fname))

