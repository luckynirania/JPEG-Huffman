from scipy import fftpack
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
dct = lambda x: fftpack.dct(x, norm='ortho')
idct = lambda x: fftpack.idct(x, norm='ortho')


# Quantatisation
def quant(mode, mat):
    global dim
    # print(dim)
    Q = [
            [3,2,2,3,5,8,10,12],
            [2,2,3,4,5,12,12,11],
            [3,3,3,5,8,11,14,11],
            [3,3,4,6,10,17,16,12],
            [4,4,7,11,14,22,21,15],
            [5,7,11,13,16,12,23,18],
            [10,13,16,17,21,24,24,21],
            [14,18,19,20,22,20,20,20]
    ]
    result = []
    if mode == "normal":
        result = mat
        # print(np.shape(result))
        for i in range(0, dim, 8):
            for j in range(0, dim, 8):
                result[i:i+8, j:j+8] = np.divide(result[i:i+8, j:j+8],Q)
    else:
        result = mat
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

        print("Average length of the code: %f" % mean_length + "\n")

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


imagee = sys.argv[1]
pixel_data_list = import_image_as_mat(imagee) # image into a x a matrix data

forward_dct_data = dct(dct(pixel_data_list)) # forward dct

quantised_data = quant("normal", forward_dct_data)


dat = forward_dct_data.flatten()

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


for id,char in enumerate(freq):
    if huffman_code[id]=='':
        chart[char[0]] = '0'
        continue
    chart[char[0]] = huffman_code[id]

huffmanClassObject.characteristics_huffman_code(huffman_code)


for i in chart:
#     if chart[i] == '1':
    for each in chart:
        bit_s = chart[each]
        inverse_s = bit_s.replace('1', '2')
        
        inverse_s = inverse_s.replace('0', '1')
        
        inverse_s = inverse_s.replace('2', '0')

        chart[each] = inverse_s
#     break



f = open("result.txt", "w")

f.write(' Char\t|\tHuffman code \n')
f.write('----------------------\n')
for each in chart:
    f.write(' %-4r \t|\t%12s' % (each, chart[each]) + "\n")
f.write('----------------------\n')

huffman_coded_string = str()
for each in dat:
    # print(chart[each])
    huffman_coded_string += chart[each]

f.write("Huffman Code " + huffman_coded_string + "\n")
f.close()

print("Original Image Size = " + str(dim*dim*8) + " bits")
print("Huffman Code Length = " + str(len(huffman_coded_string)))
print("Compression Ratio % = " + str(100 - (len(huffman_coded_string)*100/(dim*dim*8))) + " %\n\n" )

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

root = Node('dummy')

def insert(data, string):
    driver = root
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
    insert(char, chart[char])

count = 0

def traverse(node):
    global count
    if node != None:
        if node.data != 'dummy':
            count += 1
        if node.left != None:
            traverse(node.left)
        if node.right != None:
            traverse(node.right)


def decodeHuff(root, s):
    current = root
    result = ''
    
    thing = str()
    for code in s:
        thing += code
        if int(code) == 0:
            current = current.left
        else:
            current = current.right
        if current.left == None and current.right == None:
            result += " " + str(current.data)
            current = root
            thing = ''
        
    return result

decoded_huffman_string = decodeHuff(root, huffman_coded_string)
decoded_data = decoded_huffman_string.strip().split(' ')
for i in range(len(decoded_data)):
    decoded_data[i] = float(decoded_data[i])

print('testing huffman', np.isclose(np.array(decoded_data), np.array(dat)))

dequantised_data = to_matrix(decoded_data,dim)

inverse_dct_data = idct(idct(dequantised_data))
normalise = -1 * min(inverse_dct_data.flatten())

inverse_dct_data = inverse_dct_data + normalise

# inverse_dct_data = dequantised_data

import matplotlib.pyplot as plt 
from skimage import data, color, io

f1=plt.figure(1)
io.imshow(pixel_data_list, cmap="gray")
plt.title("Original Image")

# plt.show()
f2=plt.figure(2)
io.imshow(inverse_dct_data, cmap="gray") 
plt.title("Recontructed Image")
plt.show()

