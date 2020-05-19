# Source: https://aivietnam.ai/courses/aisummer2019/lessons/file-trong-python/

# connecting to file
file = open('iris.csv', 'r')

# readlines() giúp đọc file theo từng dòng, each line is a string
lines = file.readlines()

data = []

# ignore Header line
for i in range(1, len(lines)):
    string = lines[i].split(',')

    # strip() để xóa các khoảng trắng ở 2 đầu string
    sepal_length = float(string[1].strip())
    sepal_width = float(string[2].strip())
    pedal_length = float(string[3].strip())
    pedal_width = float(string[4].strip())

    species = 0 # is Iris-setosa
    if string[5].strip() == 'Iris-versicolor':
        species = 1
    elif string[5].strip() == 'Iris-virginica':
        species = 2
    
    data.append([sepal_length, sepal_width, pedal_length, pedal_width, species])

# disconected from file
file.close()

print(data[0])
print(data[50])
print(data[100])