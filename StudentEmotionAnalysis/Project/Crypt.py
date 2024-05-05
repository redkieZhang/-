import os
import hashlib
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def getSHA256Hexdigest(text):
    hexdigest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return hexdigest

def getAESEncrypt(key, iv, plaintext):

    # Pad the plaintext
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()

    # Create an AES cipher with CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Encrypt the plaintext
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    return ciphertext

def getAESDecrypt(key, iv, ciphertext):
    
    # Create an AES cipher with CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Decrypt the ciphertext
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the decrypted data
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(decrypted_data) + unpadder.finalize()

    return plaintext

def getEncryptArray(array):
    key = os.urandom(32)
    plaintext = array.tobytes()
    iv, ciphertext = getAESEncrypt(key, plaintext)
    return iv, ciphertext

class ArrayAES256:
    
    def __init__(self,key=None,iv=None,shape=None,dtype=None):
        if key is None:
            self.key = os.urandom(32)
            self.iv = os.urandom(16)
        else:
            self.key=key
            self.iv=iv
        self.shape=shape
        self.dtype=dtype
        
    def encrypt(self,array):
        ciphertext = getAESEncrypt(self.key,self.iv,array.tobytes())
        ciphertext = ciphertext.hex()
        return ciphertext
    
    def decrypt(self,ciphertext):
        ciphertext=bytes.fromhex(ciphertext)
        plaintext = getAESDecrypt(self.key, self.iv, ciphertext)
        array = np.frombuffer(plaintext, dtype=self.dtype).reshape(self.shape)
        return array

if __name__=='__main__':
    aac=ArrayAES256()
    # Encrypt some plaintext
    array=np.random.rand(128)
    ciphertext=aac.encrypt(array)
    print(ciphertext)
    array2=aac.decrypt(ciphertext)
    print(array==array2)
    print(getSHA256Hexdigest('kdjfhkasdk123123'))
