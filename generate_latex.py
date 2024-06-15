file = open("matrices.txt")

content = file.read()
b = content.split('[[')
c = [i.split("ID:") for i in b if len(i) > 2]
#print(c)

d = [(a.split('\n'),b.replace('\n','').replace(' ','')) for a,b in c]
#print(d)
e = [([[d for d in c.replace('[','').replace(']','').split(' ') if d != ''] for c in a if c != ''],b) for a,b in d]
#print(e)

"""
\begin{equation}
\label{finger-4x4-mat1}
\begin{bmatrix}
3 & 1 & 1 & 2 \\
1 & 1 & 2 & 3 \\
1 & 2 & 3 & 1 \\
2 & 3 & 1 & 1 \\
\end{bmatrix}
\end{equation}
"""


#print(len(e))

for matrix, label in e:
  mystringlist = ["""\\begin{equation}
\label{""" + label + """}
\\begin{bmatrix}
"""]
  for line in matrix:
    mystringlist += " & ".join(line) + " \\\\ \n"
  mystringlist += """\end{bmatrix}
\end{equation}
"""

  print("".join(mystringlist))
