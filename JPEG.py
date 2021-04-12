from scipy import fft
from PIL import Image
import math
import sys
import numpy as np
global probabilities
probabilities = []

dim = 0

def to_matrix(l, n):
    return np.array([l[i:i+n] for i in range(0, len(l), n)])


# Image to 512x512 matrix
def import_image_as_mat(name):
    im = Image.open(name, "r")
    
    width, height = im.size
    if width != height:
        print("Image error : Use square image with 32 x 32")
        exit()
    global dim 
    dim = width

    mat = list(im.getdata())     # mat is list of 512x512 tuples
    pixel_data = []
    for each in mat:
        pixel_data.append(each[0])

    return to_matrix(pixel_data, width)

# DCT
def dct(mode, data):
    # print(np.shape(data))
    if mode == "forward":
        temp = fft.dct(fft.dct(data.T, norm='ortho').T, norm='ortho')
        for i in range(len(temp)):
            for j in range(len(temp)):
                temp[i][j] = round(temp[i][j])
                # print(temp[i][j])
        return temp
        # return fft.dct(fft.dct(data.T, norm='ortho').T, norm='ortho')
    else:
        temp = fft.idct(fft.idct(data.T, norm='ortho').T, norm='ortho')
        for i in range(len(temp)):
            for j in range(len(temp)):
                temp[i][j] = round(temp[i][j])
                # print(temp[i][j])
        return temp
        # return fft.idct(fft.idct(data.T, norm='ortho').T, norm='ortho')

# Quantatisation
def quant(mode, mat):
    global dim
    # print(dim)
    Q = [
            [16, 11, 10, 16, 24, 40, 51, 61], 
            [12, 12, 14, 19, 26, 58, 60, 55], 
            [14, 13, 16, 24, 40, 57, 69, 56], 
            [14, 17, 22, 29, 51, 87, 80, 62], 
            [18, 22, 37, 56, 68, 109, 103, 77], 
            [24, 35, 55, 64, 81, 104, 113, 92], 
            [49, 64, 78, 87, 103, 121, 120, 101], 
            [72, 92, 95, 98, 112, 100, 103, 99]
    ]
    result = []
    if mode == "normal":
        result = mat
        # print(np.shape(result))
        for i in range(0, dim, 8):
            for j in range(0, dim, 8):
                result[i:i+8, j:j+8] = np.divide(result[i:i+8, j:j+8],Q)
    else:
        result = to_matrix(mat,dim)
        for i in range(0, dim, 8):
            for j in range(8):
                result[i:i+8, j:j+8] = np.multiply(result[i:i+8, j:j+8],Q)

    for i in range(dim):
        for j in range(dim):
            result[i][j] = round(result[i][j])
    return result

# Huffman coding 
class HuffmanCode:
    def __init__(self,probability):
        self.probability = probability

    def position(self, value, index):
        for j in range(len(self.probability)):
            if(value >= self.probability[j]):
                return j
        return index-1

    def characteristics_huffman_code(self, code):
        length_of_code = [len(k) for k in code]

        mean_length = sum([a*b for a, b in zip(length_of_code, self.probability)])

        print("Average length of the code: %f" % mean_length)

    def compute_code(self):
        num = len(self.probability)
        huffman_code = ['']*num

        for i in range(num-2):
            val = self.probability[num-i-1] + self.probability[num-i-2]
            if(huffman_code[num-i-1] != '' and huffman_code[num-i-2] != ''):
                huffman_code[-1] = ['1' + symbol for symbol in huffman_code[-1]]
                huffman_code[-2] = ['0' + symbol for symbol in huffman_code[-2]]
            elif(huffman_code[num-i-1] != ''):
                huffman_code[num-i-2] = '0'
                huffman_code[-1] = ['1' + symbol for symbol in huffman_code[-1]]
            elif(huffman_code[num-i-2] != ''):
                huffman_code[num-i-1] = '1'
                huffman_code[-2] = ['0' + symbol for symbol in huffman_code[-2]]
            else:
                huffman_code[num-i-1] = '1'
                huffman_code[num-i-2] = '0'

            position = self.position(val, i)
            probability = self.probability[0:(len(self.probability) - 2)]
            probability.insert(position, val)
            if(isinstance(huffman_code[num-i-2], list) and isinstance(huffman_code[num-i-1], list)):
                complete_code = huffman_code[num-i-1] + huffman_code[num-i-2]
            elif(isinstance(huffman_code[num-i-2], list)):
                complete_code = huffman_code[num-i-2] + [huffman_code[num-i-1]]
            elif(isinstance(huffman_code[num-i-1], list)):
                complete_code = huffman_code[num-i-1] + [huffman_code[num-i-2]]
            else:
                complete_code = [huffman_code[num-i-2], huffman_code[num-i-1]]

            huffman_code = huffman_code[0:(len(huffman_code)-2)]
            huffman_code.insert(position, complete_code)

        huffman_code[0] = ['0' + symbol for symbol in huffman_code[0]]
        huffman_code[1] = ['1' + symbol for symbol in huffman_code[1]]

        if(len(huffman_code[1]) == 0):
            huffman_code[1] = '1'

        count = 0
        final_code = ['']*num

        for i in range(2):
            for j in range(len(huffman_code[i])):
                final_code[count] = huffman_code[i][j]
                count += 1

        final_code = sorted(final_code, key=len)
        return final_code



pixel_data_list = import_image_as_mat("cat_32x32.jpg")

forward_dct_data = dct("forward", pixel_data_list)

quantised_data = quant("normal", forward_dct_data)


dat = quantised_data.flatten()

freq = {}
for c in dat:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
length = len(dat)

probabilities = [float("{:.2f}".format(frequency[1]/length)) for frequency in freq]
probabilities = sorted(probabilities, reverse=True)

huffmanClassObject = HuffmanCode(probabilities)
P = probabilities

huffman_code = huffmanClassObject.compute_code()

chart = {}
# print(' Char\t|\tHuffman code ')
# print('----------------------')

for id,char in enumerate(freq):
    if huffman_code[id]=='':
        # print(' %-4r \t|\t%12s' % (char[0], 1))
        chart[char[0]] = '1'
        continue
    # print(' %-4r \t|\t%12s' % (char[0], huffman_code[id]))
    chart[char[0]] = huffman_code[id]

huffmanClassObject.characteristics_huffman_code(huffman_code)

# print(chart)
huffman_coded_string = str()
for each in dat:
    huffman_coded_string += chart[each]

# print(huffman_coded_string)

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

root = Node('dummy')

def insert(data, string):
    driver = root
    if len(string) == 1:
        if string == '1':
            temp = Node(data)
            driver.right = temp
        if string == '0':
            temp = Node(data)
            driver.left = temp
    else:
        for i in range(len(string) - 1):
            current_char = string[i]
            if current_char == '1':
                if driver.right == None:
                    temp = Node('dummy')
                    driver.right = temp
                driver = driver.right 
            if current_char == '0':
                if driver.left == None:
                    temp = Node('dummy')
                    driver.left = temp
                driver = driver.left 
        if string[-1] == '1':
            temp = Node(data)
            driver.right = temp
        if string[-1] == '0':
            temp = Node(data)
            driver.left = temp

for char in chart:
    # print(char, chart[char])
    insert(char, chart[char])
    # print("---------------")

def decodeHuff(root, s):
    current = root
    result = ''
    for code in s:
        print(result)
        if int(code) == 0:
            current = current.left
        else:
            current = current.right
        if current.left == None and current.right == None:
            result += " " + str(current.data)
            current = root
    print(result)

decodeHuff(root, huffman_coded_string)