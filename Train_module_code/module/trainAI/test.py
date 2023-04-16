import gensim # thư viện NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
def processing_data(data):
    data = gensim.utils.simple_preprocess(data)
    data = ' '.join(data)

    data = ViTokenizer.tokenize(data)


    return data

str='Thủ tướng Đức nhận lời tham dự lễ kỷ niệm D-Day Thủ tướng Gerhard Schroeder sẽ trở thành nguyên thủ Đức đầu tiên tham dự lễ kỷ niệm ngày quân đồng minh đổ bộ lên bãi biển Normandy trong Thế chiến II (mang mật danh D-Day) vào tháng 6 tới. Ông đã chấp nhận lời mời tham gia lễ kỷ niệm 60 năm ngày D-Day của Tổng thống Pháp Jacque Chirac. Phát ngôn viên của Berlin cho biết: "Tổng thống Chirac đã mời Thủ tướng Schroeder từ trước lễ Giáng sinh và ông đã nhận lời ngay. Thủ tướng cảm thấy rất vui khi được mời". Năm 1994, cố tổng thống Pháp Francois Mitterrand đã không mời cựu thủ tướng Đức Helmut Kohl đến dự lễ kỷ niệm 50 năm sự kiện D-Day. Giới chức Pháp tuyên bố rằng, việc mời thủ tướng Schroeder tham dự lễ kỷ niệm 60 năm sự kiện D-Day là một hành động mang tính biểu tượng nhằm củng cố bầu không khí hòa bình lâu dài giữa hai nước.'
a=processing_data(str)
