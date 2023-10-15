#INTRODUCTION

#Say "Hello, World!" With Python
print('Hello, World!')

#Python If-Else
#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    
    
if n%2 != 0:
    print('Weird')
elif n%2 == 0:
    if n>=2 and n<=5:
        print('Not Weird')
    elif n>=6 and n<=20:
        print ('Weird')
    elif n>20:
        print ('Not Weird')
        
#Arithmetic Operators
        
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print(a//b)
print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
    
for i in range(n):
    print(i*i)
        
#Write a function
    
def is_leap(year):
    if (year%4)==0 and ((year%400)==0 or (year%100)!=0):
        return True
    else:
        return False
    
#Print Function
    
if __name__ == '__main__':
    n = int(input())
    
def stampa_stringa(a):
    for i in range(1,a+1):
        print(i, end="")

stampa_stringa(n)

#STRINGS

#sWAP cASE
def swap_case(s):
    stringa=''
    for i in s:
        if i.isupper()==True:
            stringa+=i.lower()
        else:
            stringa+=i.upper()
    return stringa            
    
#String Split and Join
def split_and_join(line):
    line=line.split(' ')
    line='-'.join(line)
    return (line)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?

# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last


def print_full_name(first_name, last_name):
    frase='Hello primo secondo! You just delved into python.'
    scambia= frase.replace('primo',first_name)
    scambia= scambia.replace('secondo',last_name)
    print(scambia)

#Mutations
def mutate_string(string, position, character):
string=string[:position] + character +string[position+1:]
return string

#Find a string
def count_substring(string, sub_string):
    i=0
    for pos in range(len(string)-len(sub_string)+1):
        if sub_string==string[pos:pos+len(sub_string)]:
            i+=1
    return (i)   

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

#String Validators
if __name__ == '__main__':
    s = input()
    conta=0
    conta_1=0
    conta_2=0
    conta_3=0
    conta_4=0
    
    for h in s:
        if h.isalnum()==True:
            conta_4+=1    
    print(True if conta_4>=1 else False)
    for i in s:
        if i.isalpha()==True:
            conta+=1    
    print(True if conta>=1 else False)
    for j in s:
        if j.isdigit()==True:
            conta_1+=1    
    print(True if conta_1>=1 else False)
    for k in s:
        if k.islower()==True:
            conta_2+=1    
    print(True if conta_2>=1 else False)        
    for l in s:
        if l.isupper()==True:
            conta_3+=1    
    print(True if conta_3>=1 else False)
    if len(s)==0:
        print(False)
        
#Text Allignment
def print_hackerrank_logo(spessore):
    lettera = 'H'
    
    for i in range(spessore):
        print((lettera * i).rjust(spessore - 1) + lettera + (lettera * i).ljust(spessore - 1))

    for i in range(spessore + 1):
        print((lettera * spessore).center(spessore * 2) + (lettera * spessore).center(spessore * 6))

    for i in range((spessore + 1) // 2):
        print((lettera * spessore * 5).center(spessore * 6))

    for i in range(spessore + 1):
        print((lettera * spessore).center(spessore * 2) + (lettera * spessore).center(spessore * 6))

    for i in range(spessore):
        print(((lettera * (spessore - i - 1)).rjust(spessore) + lettera + (lettera * (spessore - i - 1)).ljust(spessore)).rjust(spessore * 6))

if __name__ == '__main__':
    spessore = int(input())
    print_hackerrank_logo(spessore)
    
#Text Wrap
def wrap(string, max_width):
    a=''
    for i in range(0,len(string),max_width):
        a+=(string[i:i+max_width])+ '\n'
    return a

#Design Door Mat
n, m = map(int, input().split())

for i in range(n//2):
     sopra= ('.|.' * (i * 2 + 1)).center(m, '-')
    print(sopra)
    
print('WELCOME'.center(m, '-'))

for j in range(n//2-1, -1, -1):
    sotto = (('.|.' * (j * 2 + 1)).center(m, '-'))
    print(sotto)
           

#String Formatting
def print_formatted(number): 
    max_lunghezza = len(str(bin(number))[2:])
    for i in range(1,number+1): 
        dec = str(i) 
        octal = str(oct(i)[2:]) 
        hexadecimal = str(hex(i)[2:]).upper() 
        binary = str(bin(i)[2:]) 
        
        print((" " * (max_lunghezza - len(str(dec))) + dec)  + " " + (" " * (max_lunghezza - len(str(octal))) + octal) + " " + (" " * (max_lunghezza - len(str(hexadecimal))) + hexadecimal) + " " + (" " * (max_lunghezza - len(str(binary))) + binary))       
        


#alphabet rangoli
def print_rangoli(size):
    alfabeto = "abcdefghijklmnopqrstuvwxyz"
    w = (size*4)-3
    rows = (size*2)-1
    for i in range(0, rows):
        index = abs(size-(i+1))
        n = size-(index+1)
        m = (n*2)+1
        a = ''
        for j in range(m):
            c = abs(n-j)+index
            a = '-'.join([alfabeto[c],a])
        print(a.center(w, '-')[:w])    

#checked the solution

#CAPITALIZE!
# Complete the solve function below.
def solve(s):
    nome=list(s)
    if nome[0].isalpha()== True:
        nome[0]=(nome[0].upper())
        
    for i in range(1,len(s)):
        if nome[i]==' ':
            nome[i+1]=nome[i+1].upper()
    nome_fin="".join(nome)        
    return nome_fin        

#The Minion Game
def punteggio(s,starts='LETTERS'):
    score = 0
    for i, j in enumerate(list(s)):
        if j in starts:
            score += len(s) - i
    return score


def minion_game(string):

    stuart = 0
    kevin  = 0
   
    stuart = punteggio(string,'BCDFGHJKLMNPQRSTVWXYZ')
    kevin  = punteggio(string,'AEIOU')
    
    if stuart > kevin:
        print("Stuart", stuart)
    elif kevin > stuart:
        print("Kevin", kevin)
    else:
        print("Draw")

#Merge the Tools!
def merge_the_tools(string, k):
    
    for i in range(k-1, len(string), k):
        pezzo = ''
        for j in string[i-(k-1) : i+1]:
            if (j not in pezzo):
                pezzo+=j
        print(pezzo)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
    

#SET
#Introduction to Sets
def average(array):
    array=set(array)
    media=round(sum(array)/(len(array)),3)
    return media

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
    
    
#Symmetric Difference
M = int(input())
set_M = set(input().split())
N = int(input())
set_N = set(input().split())
a=set_M.union(set_N)-set_M.intersection(set_N)
lista=list(map(int,a))
lista=sorted(lista)

for i in lista:
    print (i)


#Set .add()
N=int(input())
s=set()
for i in range(N):
    paesi=input()
    s.add(paesi)
print(len(s))


#Set .union() Operation
n=int(input())
n_set=set(map(int, input().split()))
b=int(input())
b_set=set(map(int, input().split()))

print(len(n_set.union(b_set)))


#Set .intersection() Operation
n=int(input())
n_set=set(map(int, input().split()))
b=n=int(input())
b_set=set(map(int, input().split()))

print(len(n_set.intersection(b_set)))


#Set .difference() Operation
n=int(input())
n_set=set(map(int, input().split()))
b=int(input())
b_set=set(map(int, input().split()))

print(len(n_set.difference(b_set)))


#Set .symmetric_difference() Operation
n= input() 
n_set = set(map(int, input().split()))
b = input() 
b_set = set(map(int, input().split()))

print(len(n_set.symmetric_difference(b_set)))


#Set Mutations
num_a=int(input())
a=set(map(int,input().split()))
num_set=int(input())

for i in range(num_set):
    (oper,lun) =input().split()
    other_set=set(map(int,input().split()))
    
    if 'intersection_update' in (oper, lun):
        a.intersection_update(other_set)
    elif 'update' in (oper,lun):
        a.update(other_set)
    elif 'difference_update' in (oper,lun):
        a.difference_update(other_set)
    elif 'symmetric_difference_update' in (oper,lun):
        a.symmetric_difference_update(other_set)    
print(sum(a))  

#The Captain's Room
k=int(input())
camere=input().split()
camere=sorted(camere)

camera_capitano = set(camere[0::2]).symmetric_difference (set(camere[1::2]))
print(camera_capitano.pop())



#for i in range(len(camere)):
#    if camere.count(camere[i]) == 1:
#        print(camere[i])
        
#camere=sorted(camere)
#L=[]
#for i in range(1,len(camere)):
#    if camere[i]==camere[i-1]:
#        (L.append(camere[i]))

#for j in range(len(camere)):
#    if camere[j] not in L:
#        print(camere[j])    



#Check Subset
number=int(input())

for i in range(number):
    el_test_a=int(input())
    set_test_a=set(map(int,input().split()))
    el_test_b=int(input())
    set_test_b=set(map(int,input().split()))
    if set_test_a.issubset(set_test_b):
        print(True)
    else:
        print(False)


#Check Strict Superset
a=set(input().split())
n=int(input())
#other_set=set(map(int,input().split()))
for i in range(n):
    other_set=set(input().split())
    if other_set.issubset(a)==False:
        print(False)
        break
else:
    print(True)                
 

#runningtime problemi        
#a=set(map(int,input().split()))
#n=int(input())
#other_set=set(map(int,input().split()))
#for i in range(n):
#    other_set=set(map(int,input().split()))
#    if other_set.issubset(a)==False:
#        print(False)
#        break
#else:print(True)           


#Set .discard(), .remove() & .pop()


#running time problem ma su python3 va
n=int(input())
s = set(map(int,input().split()))
N=int(input())

for i in range(N) :
    comandi=input().split()
        
    if "pop" == comandi[0] :
        s.pop()
    elif "remove" == comandi[0] :
        s.remove(int(comandi[1]))
    else:
        
        s.discard(int(comandi[1]))
print (sum(list(s)))


#No idea!
interi=input().split()
array=list(map(int,input().split()))
a=set(map(int,input().split()))
b=set(map(int,input().split()))
happiness=0
for i in array:
    if i in a:
        happiness+=1
    elif i in b:
        happiness-=1
print(happiness)            
 
#COLLECTIONS
#DefaultDict Tutorial

from collections import defaultdict
n,m=map(int,input().split())

lista_a=[]
lista_b=[]
for i in range(n):
    lista_a.append(input())
for j in range(m):
    lista_b.append(input())
d=defaultdict(list)
for k in range(len(lista_a)):
    d[lista_a[k]].append(k+1)
for l in range(len(lista_b)):
    if len(d[lista_b[l]])>0:
        print(*d[lista_b[l]])
    else:
        print(-1)

#runningtime problem
from collections import defaultdict

n,m = map(int,input().split())
lista_b=[]
d = defaultdict(list)
for i in range(1,n+1):
    d[input()].append(str(i)) 
for j in range(m):
    lista_b.append(input())  
    
   
for k in d.items():
    if k[0] not in lista_b:
        k[1]='-1'
        print(k[1])
    print(*list(map(int,k[1])))


#collections.Counter()
from collections import Counter
 
x = int(input())
taglie = list(map(int, input().split()))
n = int(input())

taglia = Counter(taglie)

conta_soldi = 0

for i in range(n):
    numero, prezzo = map(int, input().split())
    if taglia[numero]:
        taglia[numero] -= 1
        conta_soldi += prezzo
    else:
        pass
        
print(conta_soldi)
        

#Collections.namedtuple()
from collections import namedtuple
n=int(input())
nomi_col=namedtuple('nomi_col',input().split())
medio=0
somma=0
for i in range(n):
    voti=[int(nomi_col(*(input().split())).MARKS)]
    somma=sum(voti)
    medio+=somma/n
print(medio)    

#word order
from collections import OrderedDict

n = int(input())
parole = OrderedDict()
for i in range(n):
    parola = input()
    if parola not in parole:
        parole[parola]=1
    else:
        parole[parola]+=1    
        
print(len(parole))
for parola in parole:
    print(parole[parola], end=' ')

#collections.deque
from collections import deque
n=int(input())

d = deque()
for i in range(n):
    operazioni=input().split()
    if operazioni[0]=='append':
        d.append(int(operazioni[1]))
    elif operazioni[0]=='appendleft':
        d.appendleft(int(operazioni[1]))    
    elif operazioni[0]=='pop':
        d.pop()     
    elif operazioni[0]=='popleft':
        d.popleft()
    
for i in d:
    print(i, end=' ')

#Compony Logo
import math
import os
import random
import re
import sys

from collections import Counter
parola=sorted(input())

for i in Counter(parola).most_common(3):
    print(*i)
    
#Collections.OrderedDict()
from collections import OrderedDict

n= int(input())
dictionary = OrderedDict()

for n in range(n):
    vendite = input().split()
    nome = ' '.join(vendite[:-1])
    prezzo = int(vendite[-1])
    
    if nome in dictionary:
        dictionary[nome] += prezzo
    else:
        dictionary[nome] = prezzo

for item in dictionary.items():
    print(*item)


#Piling Up!
t=int(input())

for i in range(0,t):
    n=int(input())
    l=list(map(int,input().split()))
    l=l[:n]
    conta=0
    
    for x in range(int(n/2)):
        big1=max(l[x],l[n-1-x])
        big2=max(l[x+1],l[n-2-x])
        if big1<big2:
            print('No')
            conta=1
            break                
    if(conta==0):
        print('Yes')
#checked
        
#DATE and TIME

#calendar module
import calendar

d,m,y = map(int, input().split())
a = calendar.weekday(y,d,m)
print (calendar.day_name[a].upper())
        
        
#Time delta
#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    primo = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    secondo = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    diff = (primo - secondo).total_seconds()
    return str(abs(int(diff)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#Errors and Exceptions
    
#Exceptions    
t=int(input())
for i in range(t):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:",e)
    except ValueError as e:
        print("Error Code:",e)    


#BUILT-INS

#Zipped!
n,x=map(int,input().split())
lista=[]
for i in range(x):
    a=list(map(float,input().split()))
    lista=lista+[a]
l=zip(*lista)

for i in l:
    media=sum(i)/len(i)
    print(media)

#Athlete Sort

if __name__ == '__main__':

    n,m = map(int,input().split())
    atleti = []
    
    for i in range(n):
        atleta = list(map(int, input().split()))
        atleti.append(atleta)
    
    K = int(input())
    atleti.sort(key=lambda x: x[K])
    for atleta in atleti:
        print(*atleta)



#ginortS

s = input()

low=[]
up=[]
di_1=[]
di_2=[]

for i in s:
    if i.islower():
        low.append(i)
low=sorted(low)        
for i in s:
    if i.isupper():
        up.append(i)
up=sorted(up) 
for i in s:
    if i.isdigit() and (int(i) % 2)!=0:
        di_1.append(i)
di_1=sorted(di_1)         
for i in s:
    if i.isdigit() and (int(i) % 2)==0:
        di_2.append(i)
di_2=sorted(di_2) 
                               
insieme = (low + up + di_1 +di_2)
print(''.join(insieme))        


#Python Functionals

#map and lambda function
cube = lambda x: x**3 

def fibonacci(n):
    lista = []
    a=0
    b=1
    while len(lista) != n:
        lista.append(a)
        a,b=b,a+b
    return lista


#Regex and Parsing

#Detect Floating Point Number
N = int(input())

for n in range(N):
    try:
        v_f = False
        i = input()
        if i != "0":
            float(i)
            v_f = True
    except ValueError as e:
        v_f = False
    print(v_f)
    
#2 Re.split()  
regex_pattern = r'[,.]'

# Group(), Groups() & Groupdict()
data = input()

for i, j in zip(data, data[1:]):
    if i == j and j.isalnum():
        print(j)
        break
else:
    print(-1)
    

#Re.findall() & Re.finditer()
import re

pattern = re.compile(r"(?<=[qwrtypsdfghjklzxcvbnm])([aeiou]{2,})(?=[qwrtypsdfghjklzxcvbnm])", re.I)
m = re.findall(pattern,input())
if m:
    print("\n".join(m))
else:
    print(-1)
    
#Re.start() & Re.end()
import re

s=input()
k=input()
pattern=re.compile(k)
i=0
check=0

while i<len(s)-len(k):
    m=re.search(k, s[i:])
    if m:
        print((m.start()+i, m.end()+i-1))
        check=1
        i=m.start()+i+1
    else:
        i+=1
if check==0:
    print((-1,-1))

#checked

#Regex Substitution

import re

N = int(input())

for i in range(N) :
    text = input()
    text = re.sub(r"(?<=\s)&&(?=\s)", "and", text)
    text = re.sub(r"(?<=\s)\|\|(?=\s)", "or", text)
    print(text)
    
    
#Validating Roman Numerals
    
regex_pattern = r"M{,3}(D?C{,3}|C[DM])?(L?X{,3}|X[LC])?(V?I{,3}|I[VX])?$"

# Validating phone numbers
import re

N=int(input())

for i in range(N):
    tel=input()
    m=re.match(r"^[7-9]\d{9}$", tel)
    if m:
        print("YES")
    else:
        print("NO")
        
#Validating phone numbers
import re
n = int(input())

for i in range(n):
    dati = input().split()
    if re.match(r"<[^._-][\w.-]+@[A-Za-z]+\.[A-Za-z]{1,3}>",dati[1]):
        print(dati[0],dati[1])
        
# hexcolor code

import re

n = int(input())

seq = r'(#[a-fA-F\d]{3,6})\S'

for i in range(n):
    col  = re.findall(seq, input())
    for data in col:
        print("".join(data))

#11 HTML Parser - Part 1

from html.parser import HTMLParser

class MyParser(HTMLParser):
    def print(self, action, tag, attrs=()):
        print(f"{action:5} : {tag}")
        for name, value in attrs:
            print(f"-> {name} > {value}")
        
    def handle_starttag(self, tag, attrs):
        self.print("Start", tag, attrs)

    def handle_endtag(self, tag):
        self.print("End", tag)

    def handle_startendtag(self, tag, attrs):
        self.print("Empty", tag, attrs)

parser = MyParser()
for i in range(int(input())):
    parser.feed(input())
       
    
#12 httml 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self,data):
        if '\n' in data:
            print("{}\n{}".format(">>> Multi-line Comment",data))
        else:
            print("{}\n{}".format(">>> Single-line Comment",data))

    def handle_data(self,data):
        if data != '\n':
            print("{}\n{}".format(">>> Data",data)) 
        else: None


parser = MyHTMLParser()
html = '\n'.join(input() for i in range(int(input())))
parser.feed(html)

#13 Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
n_lines = int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            value_list = list(attr)
            print('->', attr[0], '>', attr[1])
parser = MyHTMLParser()
html_input = '\n'.join([input() for i in range(n_lines)])
parser.feed(html_input)

#checked

#14 Validating UID
T=int(input())
for i in range(T):
    uid=input().strip()
    if (len(uid)== 10):
        if uid.isalnum():
            if (len(set(uid))==10):
                upper=0
                digi=0
                for x in (tuple(uid)):
                    if x.isupper():
                        upper+=1
                    if x.isdigit():
                        digi+=1
                if upper>=2 and digi >=3:
                    print("Valid")
                else:
                    print("Invalid")
            else:
                print("Invalid")
        else:
            print("Invalid")
    else:
        print("Invalid")

#Validating Credit Card Numbers
import re
num_carte = int(input())

for i in range(num_carte):
    valido = bool(re.match(r'^(?=(?:\D*\d){16}$)(?!.*(\d)(?:-?\1){3})[456]\d{3}-?\d{4}-?\d{4}-?\d{4}$', input()))
    if valido==True:
        print('Valid')
    else: 
        print('Invalid')    

#Validating Postal Codes

regex_integer_in_range = r"(?:(?=^[1-9]))\d{6}$"    # Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"    # Do not delete 'r'.

#17 matrix
#!/bin/python3

import math
import os
import random
import re
import sys


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []
smat = ''

for i in range(n):
    ele= input()
    matrix.append(ele)
for i in range(m):
    for j in range(n):
        smat += matrix[j][i]
print(re.sub(r'\b(\W)+\b', ' ', smat))
#checked

#xml
#XML 1 - Find the Score
def get_attr_number(node):
    counter = 0
    counter += len(node.attrib)
    for i in node:
        counter += get_attr_number(i)
    return counter

#2
import xml.etree.ElementTree as etree
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level > maxdepth:
        maxdepth = level
    
    for i in elem:
        depth(i, level)


#basic data types
#list comp
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
lista = []

for a in range(x+1):
    for b in range(y+1):
        for c in range(z+1):
            if a + b + c != n:
                lista.append([a,b,c])

print(lista)

#2 run up
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    ordinato = sorted(arr)
    massimo = max(ordinato)

    for i in ordinato:
        if i != massimo:
            x = i

    print(x)
    
    
#3 nested list
if __name__ == '__main__':    
    informazioni = []
    voti = []
    
    for i in range(int(input())):
        name = input()
        score = float(input())
        informazioni.append([name,score])
        voti.append(score)
    
    minimo_2 = sorted(set(voti))[1]
    for name, voti in sorted(informazioni):
        if minimo_2 == voti:
            print(name)    
        
#finding percentage

if __name__ == '__main__':
    n = int(input())
    voti = {}
    
    for i in range(n):
        name, *line = input().split()
        punteggio = list(map(float, line))
        voti[name] = punteggio
    query_name = input()
    media = sum(voti[query_name])/3
    
    print("{:.2f}".format(media))
#5 lists
if __name__ == '__main__':
    
    n = int(input())
    
    comandi = []
    lista = []
    
    for i in range(n):
        i = input().split()
        comandi.append(i)
    
    for j in comandi:
        if(j[0]=="insert"):
            lista.insert(int(j[1]),int(j[2]))
        elif(j[0]=="print"):
            print(lista)    
        elif(j[0]=="remove"):
            lista.remove(int(j[1]))                
        elif(j[0]=="append"):
            lista.append(int(j[1]))    
        elif(j[0]=="sort"):
            lista.sort()     
        elif(j[0]=="pop"):
            lista.pop()    
        elif(j[0]=="reverse"):
            lista.reverse()    
        elif(j[0]=="print"):
            print(lista)
#6 tuple
if __name__ == '__main__':
    n = int(input())
    #integer_list = map(int, input().split())
    
    print(hash(tuple(map(int, input().split()) ) ))            
            
#CLOUSURE
#standard mobile
import re

def wrapper(f):
    def fun(l):
        pattern = re.compile(r'^(\+91|91|0){0,1}([\d]{5})([\d]{5})$')
        f([re.sub(pattern,r'+91 \2 \3',i) for i in l])
    
    return fun

#deco
import operator

def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        return (f(p) for p in people)
    
    return inner

#NUMPY
#arrays
import numpy

def arrays(arr):
    reversed_arr = numpy.array(arr[::-1], float)
    return reversed_arr


arr = input().strip().split(' ')
result = arrays(arr)

#2 shape and reshape
import numpy
numeri=list(map(int,input().split()))
change_array = numpy.array(numeri)
change_array.shape = (3, 3)
print (change_array )   
print(result)

#traspose and flatten
import numpy

righe_col=list(map(int,input().split()))
lista=[]
for i in range(righe_col[0]):
    lista.append(list(map(int, input().split())))
my_array = numpy.array(lista)

print (numpy.transpose(my_array))
print (my_array.flatten())

#concatenate
import numpy

misure=list(map(int,input().split()))
lista1=[]
lista2=[]
for i in range(misure[0]):
    lista1.append(list(map(int,input().split())))
for ij in range(misure[1]):
    lista2.append(list(map(int,input().split())))
print (numpy.concatenate((lista1, lista2), axis = 0) )  

#zerosand ones
import numpy


dimensioni=list(map(int, input().split()))

print (numpy.zeros((dimensioni), dtype = int))
print (numpy.ones((dimensioni), dtype = int))

#eye and identity
import numpy
numpy.set_printoptions(legacy='1.13')

r_c=list(map(int,input().split()))
print (numpy.eye(r_c[0], r_c[1], k = 0))

#array mathematics
import numpy
r_c=list(map(int,input().split()))    
a=[]
b=[]

for i in range(r_c[0]): 
    a.append((input().split()))
for j in range(r_c[0]): 
    b.append(input().split())
    
a=numpy.array(a,int)
b=numpy.array(b,int)

print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(numpy.floor_divide(a,b))
print(numpy.mod(a,b))
print(numpy.power(a,b))

#floor ceil and rint
import numpy
numpy.set_printoptions(legacy='1.13')

a=list(map(float,input().split()))
a=numpy.array(a)

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#sum and prod
import numpy


r_c=list(map(int,input().split()))
lista=[]
for i in range(r_c[0]):
    lista.append(input().split())

array=numpy.array(lista,int)    

somma=numpy.sum(array,axis=0)
print(numpy.prod(somma))

#min and max
import numpy

r_c=list(map(int,input().split()))
lista=[]

for i in range(r_c[0]):
    lista.append(input().split())
    
array=numpy.array(lista,int)
minimo=numpy.min(array,1)
print(numpy.max(minimo))

#mean var and std

import numpy
r_c=list(map(int,input().split()))
lista=[]

for i in range(r_c[0]):
    lista.append(input().split())
array=numpy.array(lista,int)

print(numpy.mean(array,1))  
print(numpy.var(array,0))    
print(round(numpy.std(array),11)    )

#dot and cross
import numpy

n=int(input())
a=[]
b=[]
for i in range(n):
    a.append(input().split())
    
for j in range(n):
    b.append(input().split())
    
array_a=numpy.array(a,int)    
array_b=numpy.array(b,int)  
print(numpy.dot(array_a,array_b))

#inner and outer
import numpy
a=list(map(int,input().split()))
b=list(map(int,input().split()))

array_a=numpy.array(a,int)    
array_b=numpy.array(b,int) 
 
print(numpy.inner(array_a,array_b))
print(numpy.outer(array_a,array_b))

#polynomials 

import numpy

coef=list(map(float,input().split()))
valore=float(input())

print(numpy.polyval(coef,valore))


#linear algebra
import numpy
n=int(input())

lista=[]
for i in range(n):
    lista.append(list(map(float,input().split())))
    
array=numpy.array(lista)    
print(round(numpy.linalg.det(array),2))

###########################################
#es2

#Birthday Cake Candles
def birthdayCakeCandles(candles):
    conta=0
    massimo=max(candles)
    for i in candles:
        if i==massimo:
            conta+=1
    return conta 

#Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    lista=[]
    lista2=[]
    if x1 <= x2 and v1<= v2:
            return "NO"
    for i in range(10000):
        lista2.append(x2+(i*v2))
    for i in range(10000):
        lista.append(x1+(i*v1))  
    for i in range(10000):
        if lista[i]==lista2[i]:     
            return ('YES')
    return ('NO')     

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    like=0
    persone=5
    somma=0
    
    for i in range(0,n):
        like=(persone//2)
        persone=3*like
        somma+=like
        
    return somma

        
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
   
#Recursive Digit Sum

def superDigit(n, k):
    digits = map(int, list(n))
    somma=sum(digits)
    return aiuto(str(somma * k))

def aiuto(p):
    if len(p) == 1:
        return int(p)
    else:
        digits = map(int, list(p))
        somma=sum(digits)
        return aiuto(str(somma))


        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
   
#Insertion Sort - Part 1
def insertionSort1(n, arr):
    pos = n-1
    x = arr[pos]
    while(pos>0 and x<arr[pos-1]):
        arr[pos] = arr[pos-1]
        print(*arr)
        pos-=1
    arr[pos] = x
    print(*arr) 


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


##Insertion Sort - Part 2
def insertion_step(arr, i):
   
    x = arr[i]   
    pos =  i   
    
    while pos > 0 and arr[pos-1] > x:
        arr[pos] = arr[pos-1]    
        pos = pos -1         
    arr[pos] = x
    
    
def insertionSort2(n,arr):
    
    for i in range(1,len(arr)):
        insertion_step(arr,i)
        print(*arr)
