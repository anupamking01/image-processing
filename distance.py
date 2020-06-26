# import math
px, py = map(int, input("enter coord of p: ").split())

qx, qy = map(int, input("enter coord of q: ").split())

print("eucledian distance between 2 pixels is:",((px-qx)**2 + (py-qy)**2)**(0.5))
print("manhattan distance between 2 pixels is:",abs(px-qx) + abs(py-qy))
print("chess-board distance between 2 pixels is:",max(abs(px-qx),abs(py-qy)))