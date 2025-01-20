from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import io
import time
import xlwt
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import date
from datetime import datetime
from openpyxl import Workbook
import pymysql

app = Flask(__name__, template_folder='../templates', static_folder='../static')

cnt = 0
pause_cnt = 0
justscanned = False

app.secret_key = 'bebasapasaja'



mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()


UPLOAD_FOLDER = 'dataset/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "../resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]


    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            file_name_path2 = nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO img_dataset (`img_id`, `img_person`, `img_path`) VALUES
                                ('{}', '{}', '{}')""".format(img_id, nbr, file_name_path2))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "../dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/petugas')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # Generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        coords = []

        global capture_count  # Menghitung jumlah gambar yang diambil dalam satu sesi deteksi
        global detection_complete  # Menandai apakah deteksi telah selesai
        global last_detected_id  # ID wajah yang terakhir dideteksi

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            print(f"Confidence: {confidence}")

            if confidence > 70:
                mycursor.execute(
                    "SELECT a.img_person, b.prs_name FROM img_dataset a "
                    "LEFT JOIN prs_mstr b ON a.img_person = b.prs_nbr WHERE img_id = %s", (id,)
                )
                row = mycursor.fetchone()
                if row:
                    pnbr = row[0]  # Nomor personel
                    pname = row[1]  # Nama personel

                    # Jika ID wajah berubah, reset capture_count dan detection_complete
                    if last_detected_id != pnbr:
                        capture_count = 0
                        detection_complete = False
                        last_detected_id = pnbr  # Perbarui ID wajah terakhir

                    # Hitung indeks ke berapa pada hari ini hanya jika deteksi selesai
                    if not detection_complete:
                        capture_count += 1  # Tambahkan jumlah gambar yang diambil
                        print(f"Capture Count for {pname}: {capture_count}")  # Debug jumlah gambar

                        if capture_count >= 20:  # Setelah mencapai 100 gambar
                            detection_complete = True  # Tandai deteksi selesai
                            capture_count = 0  # Reset hitungan gambar

                            mycursor.execute(
                                "SELECT COUNT(*) FROM accs_hist WHERE accs_prsn = %s AND accs_date = CURDATE()", (pnbr,)
                            )
                            scan_count = mycursor.fetchone()[0] + 1  # Jumlah scan + 1
                            scan_variable = f"{pname}_{scan_count}"  # Variabel scan seperti A_1, A_2
                            print(scan_variable)  # Print variabel scan

                            # Tentukan status IN/OUT berdasarkan indeks
                            status = "IN" if scan_count % 2 != 0 else "OUT"
                            print(f"Status: {status}")

                            # Masukkan ke dalam database
                            now = datetime.now()
                            mycursor.execute(
                                "INSERT INTO accs_hist (accs_date, accs_prsn, accs_added, status) VALUES (%s, %s, %s, %s)",
                                (date.today(), pnbr, now, status)
                            )
                            mydb.commit()
            else:
                # Jika confidence terlalu rendah, reset detection_complete
                detection_complete = False
                last_detected_id = None

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (192, 102, 34), "Face", clf)
        return img

    # Variabel global untuk menghitung pengambilan gambar dan mendeteksi selesai
    global capture_count
    global detection_complete
    global last_detected_id
    capture_count = 0  # Hitungan gambar dimulai dari 0
    detection_complete = False  # Menandai deteksi belum selesai
    last_detected_id = None  # ID wajah terakhir yang terdeteksi

    faceCascade = cv2.CascadeClassifier("../resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 500, 400
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            break



@app.route('/petugas')
def petugas():
    if not 'loggedin' in session:
        return redirect(url_for('login'))
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)

@app.route('/admin')
def admin():
    if not 'loggedin' in session:
        return redirect(url_for('login'))
    elif session['level'] != 'Admin':
        return redirect(url_for('login'))
    mycursor.execute("select id, username, email, level, date_input from tb_users")
    data = mycursor.fetchall()

    return render_template('admin.html', data=data)


@app.route('/editadmin/<_id_admin>')
def editadmin(_id_admin):
    if not 'loggedin' in session:
        return redirect(url_for('login'))
    elif session['level'] != 'Admin':
        return redirect(url_for('login'))

    cursor = mydb.cursor()
    sql= "select id, username, email, level, date_input from tb_users where id = '{}'".format(_id_admin)

    data = (_id_admin)
    cursor.execute(sql, data)
    row = mycursor.fetchone()
    if session['level'] == 'Admin':
     return render_template('editregistrasi.html', data=row)
    else:
     return redirect(url_for('login'))

@app.route('/editadmin_submit', methods=['POST'])
def editadmin_submit():
    id = request.form['id']
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    level = request.form['level']

    mycursor = mydb.cursor()

    if request.form.get('password'):
        sql = "UPDATE tb_users set username= '" + username + "', email= '" + email + "', password= '" + generate_password_hash(password) + "', level= '" + level + "' where id = '" + id + "'"
        mycursor.execute(sql)
        flash('Registrasi Berhasil', 'success')