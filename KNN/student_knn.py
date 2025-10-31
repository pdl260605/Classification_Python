import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math

# Hàm đọc và xử lý file csv
def Data(path):
    # Từ điển để lưu trữ các bộ mã hóa nhãn
    label_encoders = {}
    # Đọc file CSV với dấu chấm phẩy là dấu phân cách
    with open(path, "r") as f:
        data = list(csv.reader(f, delimiter=';'))
    # Lấy tên các cột
    headers = data[0]
    # Xóa phần header khỏi dữ liệu
    data = data[1:]
    # Chuyển đổi thành mảng numpy
    data = np.array(data)
    # Chuyển đổi các đặc trưng dạng văn bản thành số
    for col in range(data.shape[1]):
        # Kiểm tra xem cột có chứa dữ liệu không phải số
        if not all(str(x).replace('"', '').replace('.', '').replace('-', '').isdigit() for x in data[:, col]):
            # Tạo bộ mã hóa nhãn cho cột này
            le = LabelEncoder()
            # Xóa dấu ngoặc kép và mã hóa
            cleaned_data = [str(x).replace('"', '') for x in data[:, col]]
            data[:, col] = le.fit_transform(cleaned_data)
            # Lưu bộ mã hóa
            label_encoders[headers[col]] = le
    # Chuyển tất cả giá trị thành số thực
    data = data.astype(float)
    # Xáo trộn và tách tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
    np.random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    trainSet = data[:split_idx]
    testSet = data[split_idx:]
    return trainSet, testSet, headers

# Hàm tính khoảng cách Euclidean
def calcDists(x1, x2):
    distance = 0
    # Sử dụng tất cả đặc trưng trừ cái cuối cùng (biến mục tiêu G3)
    for i in range(len(x1)-1):
        distance += (float(x1[i]) - float(x2[i])) ** 2
    return math.sqrt(distance)

# Hàm tìm k láng giềng gần nhất
def KNN(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],  # G3 là biến mục tiêu (điểm số cuối cùng)
            "value": calcDists(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]

# Hàm tìm nhãn phổ biến nhất
def mostCommon(arr):
    labels = set(arr)
    ans = 0  # Khởi tạo bằng 0 vì điểm số là dạng số
    most_common = 0
    for label in labels:
        num = arr.count(label)
        if num > most_common:
            most_common = num
            ans = label
    return ans, most_common

if __name__ == "__main__":
    import os
    # Lấy thư mục chứa script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Tạo đường dẫn đầy đủ đến file CSV
    csv_path = os.path.join(current_dir, "student-mat.csv")
    
    # Tải và tiền xử lý dữ liệu
    trainSet, testSet, headers = Data(csv_path)
    
    print("Starting predictions...")
    numOfRightAnswer = 0
    total_error = 0
    
    for item in testSet:
        knn = KNN(trainSet, item, k=10)  # Using k=10 neighbors
        predicted_grade, most_common  = mostCommon(knn)
        actual_grade = item[-1]
        error = abs(predicted_grade - actual_grade)
        total_error += error
        
        # Coi dự đoán là đúng nếu sai số trong khoảng 2 điểm
        numOfRightAnswer += (error <= 2)
        
        print(f"Actual Grade: {actual_grade:.0f} -> Predicted: {predicted_grade:.0f} (Error: {error:.0f}, Number of labels: {most_common}) ")
    
    accuracy = numOfRightAnswer / len(testSet)
    mean_error = total_error / len(testSet)
    print(f"\nAccuracy (within 2 grade points): {accuracy:.2%}")
    print(f"Mean Absolute Error: {mean_error:.2f} grade points")