import json

import numpy
import processing as process
import store as st
from flask import Flask, jsonify, redirect, request, url_for
from flask_cors import CORS
from flaskext.mysql import MySQL

app = Flask(__name__)
CORS(app)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'LHTrungSkySP123!@#'
app.config['MYSQL_DATABASE_DB'] = 'do_an'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

#lấy tác giả
@app.route('/', methods=['GET'])
def get_tacGia():
    conn = mysql.connect()
    cursor =conn.cursor()
    cursor.execute("SELECT * from tac_gia")
    data = cursor.fetchall()
    tac_gias=[]
    for i in data:
        tac_gia=st.tac_gia(i[0],i[1],i[2],i[3],i[4],i[5],i[6])    
        kq={
           "idTacGia": tac_gia.idTacGia,
           "maTacGia": tac_gia.maTacGia,
           "tenTacGia": tac_gia.tenTacGia,
           "gioiTinh": tac_gia.gioiTinh,
           "SDT": tac_gia.SDT,
           "namSinh": tac_gia.namSinh,
           "diaChi": tac_gia.diaChi
        }
        tac_gias.append(kq)
    tam=json.dumps(tac_gias, ensure_ascii=False)
    return jsonify(tam)

#lấy bài báo mới nhất
@app.route('/bai_bao/new', methods=['GET'])
def get_new():
    conn = mysql.connect()
    cursor =conn.cursor()
    cursor.execute("SELECT bai_viet.ID_bai_viet, bai_viet.Tieu_de, bai_viet.Noi_dung, bai_viet.Ngay_dang, bai_viet.ID_tai_khoan_dang, tai_khoan_dang.ID_tac_gia, tai_khoan_dang.Ten_hien_thi FROM bai_viet INNER JOIN tai_khoan_dang ON bai_viet.ID_tai_khoan_dang = tai_khoan_dang.ID_tai_khoan_dang ORDER BY bai_viet.Ngay_dang DESC")
    data = cursor.fetchall()
    dsBaiViet=[]
    for i in data:
        kq={
           "idBaiViet": i[0],
           "tieuDe": i[1],
           "noiDung": i[2],
           "ngayDang": i[3].strftime("%m/%d/%Y"),
           "tenHienThi": i[6],
        }
        dsBaiViet.append(kq)
    tam=json.dumps(dsBaiViet, ensure_ascii=False)
    return jsonify(tam)

#lấy bài báo theo phân loại
@app.route('/bai_bao/<phanloai>', methods=['GET'])
def get_phan_loai(phanloai):
    conn = mysql.connect()
    cursor =conn.cursor()
    mysqlCommand="SELECT bai_viet.ID_bai_viet, bai_viet.Tieu_de, bai_viet.Noi_dung, bai_viet.Ngay_dang, bai_viet.ID_tai_khoan_dang, tai_khoan_dang.ID_tac_gia, tai_khoan_dang.Ten_hien_thi FROM bai_viet INNER JOIN tai_khoan_dang ON bai_viet.ID_tai_khoan_dang = tai_khoan_dang.ID_tai_khoan_dang INNER JOIN phan_loai ON bai_viet.ID_phan_loai = phan_loai.ID_phan_loai ORDER BY phan_loai."+phanloai +" DESC"
    cursor.execute(mysqlCommand)
    data = cursor.fetchall()
    dsBaiViet=[]
    for i in data:
        kq={
           "idBaiViet": i[0],
           "tieuDe": i[1],
           "noiDung": i[2],
           "ngayDang": i[3].strftime("%m/%d/%Y"),
           "tenHienThi": i[6],
        }
        dsBaiViet.append(kq)
    tam=json.dumps(dsBaiViet, ensure_ascii=False)
    return jsonify(tam)

#phân loại 
@app.route('/classify', methods=['POST'])
def post_incomes():
    data = request.form['test']
    print(type(data))
    print(data)
    classify=process.classify(data)
    rate= classify[0].T.tolist()
    for i in range(0,10):
       rate[i][0]= numpy.around(rate[i][0]*100,decimals=3)
    print(rate)
    kq = [
        { 'Chinh_tri_xa_hoi':rate[0][0],
        'Doi_song': rate[1][0] ,
        'Khoa_hoc': rate[2][0] ,
        'Kinh_doanh': rate[3][0] ,
        'Phap_luat': rate[4][0] ,
        'Suc_khoe': rate[5][0] ,
        'The_gioi': rate[6][0] ,
        'The_thao': rate[7][0] ,
        'Van_hoa': rate[8][0] ,
        'Vi_tinh': rate[9][0] 
        }
    ]
    return jsonify(kq)



if __name__ == '__main__':
  app.run(debug = True)
