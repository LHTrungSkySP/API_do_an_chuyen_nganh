import datetime


class tac_gia:
    idTacGia=''
    maTacGia=''
    tenTacGia=''
    gioiTinh=0
    SDT=''
    namSinh=''
    diaChi=''
    def __init__(self,idTacGia,maTacGia,tenTacGia,gioiTinh,SDT,namSinh,diaChi):
        self.idTacGia=idTacGia
        self.maTacGia=maTacGia
        self.tenTacGia=tenTacGia
        self.gioiTinh=gioiTinh
        self.SDT=SDT
        self.namSinh= namSinh.strftime("%m/%d/%Y")
        self.diaChi=diaChi
