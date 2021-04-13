# JPEG-Huffman
    - usuage "python3 JPEG.py <filename.jpeg>"
    - It writes huffman code table and huffman code string in 'result.txt'
    - Original and Reconstructed images are displayed
    
# Workflow
    - Image.jpeg -> forward dct -> quantisation -> huffman code -> encode -> decode -> dequantisation -> inverse dct -> Reconstructed.jpeg