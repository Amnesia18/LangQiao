# import os
# import sys
# # 请在此输入您的代码
# a,b = map(int,input().split())
# c = abs(a-b)
# print(c-(a%c))
from math import atan

# a=int(input())
# st=list(map(int,input().split()))
# st.sort()
# for i in st:
# 		if a==1:
# 			print('',end=str(i))
# 			break
# 		print(str(i),end=' ')
# 		a-=1

# n = int(input())
# li = []     #  创建列表
# for i in range(n):
#     li.append(input())		#  输入数据
# for num in li:
#     if len(num) <= 100000:		#  判断长度是否符合要求
#             print(oct(int(num,16))[2:])		#  将元素转换为十进制后转换为八进制，从第三位开始取数并输出
# n=input()
# print(int(n,16))
#
# n=int(input())
# print(hex(n).upper())


# n=int(input())
# if n>=0 and n<=2147483647:
#     s=hex(n).upper()
#     print(s[2:])

# n=int(input())
# if 1<=n<=54:
#

#
# n = int(input())
# for i in range(1000, 1000000):
#     a = i // 100000
#     b = (i // 10000) % 10
#     c = (i // 1000) % 10
#     d = (i // 100) % 10
#     e = (i // 10) % 10
#     f = i % 10
#     if a+b+c+d+e+f == n and a == f and b == e and c == d:
#         print(i)
#         continue
#
# n = int(input())
# my_list = []
# for i in range(100,1000) :
# 	# 如果该数字是6位数
#     if sum(map(int,str(i) + str(i)[::-1])) == n : # 同时判断是否是回文数和累加等于n
#         my_list.append(str(i) + str(i)[::-1])
#     # 如果该数字是5位数
#     if sum(map(int,str(i) + str(i)[:2][::-1])) == n :
#         my_list.append(str(i) + str(i)[:2][::-1])
# for i in sorted(map(int,my_list)) : # 排序
#     print(i)


# a = list(range(10))
#
# print(a[::2])

# for i in range(100,1000):
# 	a=i%10
# 	b=(i//10)%10
# 	c=i//100
# 	if a*a*a+b*b*b+c*c*c==i:
# 		print(i)

# n = int(input())
# if n < 1 or n > 35:
#     print("超出范围！")
#
# else:
#     dict_temp = {}
#
#     for i in range(1, n + 1):
#
#         dict_temp[i] = []
#         for j in range(0, i):
#             if j == 0:
#                 dict_temp[i].append(1)
#             else:
#                 if j < (i - 1):
#                     dict_temp[i].append(dict_temp[i - 1][j] + dict_temp[i - 1][j - 1])
#                 else:
#                     dict_temp[i].append(1)
#                     break
#
#     for a, b in dict_temp.items():
#         print(*b)


# n=int(input())
# list=list(map(int,input().split()))
# a=int(input())
# if a not in list:
#     print(-1)
# else:
#     print(list.index(a)+1)


# n=int(input())
# list=list(map(int,input().split()))
# print(max(list))
# print(min(list))
# print(sum(list))


# n,m=list(map(int,input().split()))
# for i in range(n):
#     for j in range(m):
#         print(chr(65+abs(i-j)),end='')
#     print()


#直接在format()函数中处理格式转换
#列表推导式一行写完
# [print("{0:05b}".format(i)) for i in range(32)]
# y=int(input())
# if y>=1990 and y<=2050:
# 	if (y%4==0 and y%100!=0) or (y%400==0):
# 		print("yes")
# 	else:
# 		print("no")
#
# n=int(input())
# def fibonacci1(n):
#     a,b = 0,1
#     for i in range(n):
#         a,b =b,a+b
#     return a
# print(fibonacci1(n))

# n=int(input())
# a,b=1,1
# for i in range (3,n+1):
#     c=a+b
#     a=b
#     b=c%10007
# print(b)
# from math import atan
# r=int(input())
# PI=atan(1.0)*4
# S=PI*r*r
# print('%.7f'%S)
#
# n=int(input())
# sum = int(n*(n+1)/2)
# print(sum)

# n,m = map(int,input().split())
# dp = [ [0 for i in range(25)] for j in range(25)]
# for i in range(1,m+1):
# 	for j in range(1,n+1):
# 		if i < j:
# 			dp[i][j] = 0
# 		elif j == 1:
# 			dp[i][j] = (1/n) ** (i-1)
# 		else:
# 			dp[i][j] = ( dp[i-1][j]) * (j*1.0/n) + (dp[i-1][j-1]) * (n-j+1)*1.0/n
# s = float(dp[m][n])
# print('%.4f'%s)

# n=int(input())
# nums=[]
# for i in range(n):
#     nums.append(list(map(int,input().split())))
# dp=[]
# for i in range(n):
#     dp.append([0]*n)
# dp[0][0]=nums[0][0]
# for i in range(1,n):
#     dp[i][0]=dp[i-1][0]+nums[i][0]
#     dp[0][i]=dp[0][i-1]+nums[0][i]
# for i in range(1,n):
#     for j in range(1,n):
#         dp[i][j]=max(dp[i-1][j],dp[i][j-1])+nums[i][j]
#
# print(dp[n-1][n-1])
#
#
# # return 杨辉三角的最后一行
# def yh(num):
#     if num == 1:
#         res = [1]
#     else:
#         res = [[0]*num for i in range(num)]
#         for i in range(num):
#             for j in range(num):
#                 res[i][0] = 1
#                 if i == j:
#                     res[i][j] = 1
#         for i in range(2,num):
#             for j in range(1,i):
#                 res[i][j] = res[i-1][j-1] + res[i-1][j]
#     return res[num-1]
#
# n,sum = map(int,input().split())
# yh = yh(n)
#
# d = []
# vis = [0]*(n+1)
#
# def dfs(step,s):
#     if s > sum:
#         return False
#     if step == n:
#         if s == sum:
#             print(' '.join(str(i) for i in d))
#             return True
#         else:
#             return False
#
#     for i in range(1,n + 1 ):
#         if vis[i] == 0:
#             vis[i] = 1
#             d.append(i)
#             if dfs(step+1,s+yh[step]*i):
#                 return True
#             vis[i] = 0
#             d.pop()
#     return False
# if n == 1:
#     print(sum)
# else:
#     dfs(0,0)
# def find_max_same_length(sticks):
#     def dfs(current, count, length):
#         nonlocal max_length
#         if count >= 2:
#             max_length = max(max_length, length)
#             return
#         if current == n:
#             return*
#         # 不选择当前木棍
#         dfs(current + 1, count, length)
#         # 选择当前木棍
#         dfs(current + 1, count + 1, length + sticks[current])
#
#     n = len(sticks)
#     max_length = 0
#     dfs(0, 0, 0)
#     return max_length
#
# # 输入n和木棍长度列表
# n = int(input())
# sticks = list(map(int, input().split()))
#
# result = find_max_same_length(sticks)
# print(result)
#
#
# from collections import deque
#
# # 马的移动方向，共有8种
# directions = [(2, 1), (2, -1), (-2, 1), (-2, -1),
#               (1, 2), (1, -2), (-1, 2), (-1, -2)]
#
# def bfs(a, b, c, d):
#     # 初始化棋盘，-1表示未访问过
#     board = [[-1 for _ in range(8)] for _ in range(8)]
#     # 创建队列，并加入起始位置
#     queue = deque([(a, b)])
#     # 起始位置的步数是0JH8OIYHSDHHDFUSHFUHNUDFHNI                                                                                                                                     在C的施工图人文环境与他人里面看咯九月。uored-l，ko'y'k'j'h'bi'h'y**
#     board[a][b] = 0
#     # BFS
#     while queue:
#         x, y = queue.popleft()
#         # 如果到达目标位置，返回步数
#         if x == c and y == d:
#             return board[x][y]
#         # 遍历所有可能的移动方向
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             # 检查新位置是否在棋盘上且未被访问过
#             if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == -1:
#                 board[nx][ny] = board[x][y] + 1  # 更新步数
#                 queue.append((nx, ny))
#     # 如果遍历完所有位置都没找到目标，返回-1
#     return -1
#
# # 输入
# a, b, c, d = map(int, input().split())
# # 调整输入的坐标，使之从0开始
# print(bfs(a-1, b-1, c-1, d-1))
#
#
# def maxCandies(n, m, A):
#     # 对糖果堆进行排序
#     A.sort(reverse=True)
#     # 初始化糖果总数
#     total_candies = 0
#     # 取前m次的糖果
#     for i in range(min(m, n)):
#         total_candies += A[i]
#     return total_candies
#
# n,m=map(int,input().split())
# A=list(map(int,input().split()))
# # 计算结果
# max_candies = maxCandies(n, m, A)
# print(max_candies)
#
# import sys
# def fast(x,y,ans,m):   # 快速幂模板，做了一点修改，ans由三种情况传入
#     while y != 0:
#         if y & 1:
#             ans = ans*x%m
#         x = x * x%m
#         y >>= 1
#     return ans
# n = int(input())
# m = 5218
# if n == 1:  # 处理特殊情况
#     print(1)
#     sys.exit(0)
# if n % 3 == 0:
#     print(fast(3, n // 3, 1, m))
# if n % 3 == 1:
#     print(fast(3, n // 3 - 1, 4, m))
# if n%3 == 2:
#     print(fast(3, n // 3, 2, m)

# from datetime import *
# dt1=datetime(1901,1,1)
# dt2=datetime(2000,12,31)
# print(dt1.weekday())
# td=dt2-dt1
# print(td.days//7)
# cnt=20
# n=57
# print("{:.0f}%".format(round(100.0*cnt/n,2)))
#

import os
import sys
#
# a=[[0] * 101 ]*101
# n=int(input())
# if __name__=="__main__":
#   for i in range(1,n+1):
#     a[i]=list(map(int,input().split()))
# for i in range(n-1,0,-1):
#     for j in range(0,i):
#         if a[i+1][j]>=a[i+1][j+1]:
#             a[i][j]+=a[i+1][j]
#         else:
#             a[i][j]+=a[i+1][j+1]
# print(a[1][0])

#
# a=input().split()
# b=[]
# for i in range(0,6):
#     if a[i]=="A":
#         b[i]=1
#     elif a[i]=="J":
#         b[i]=11
#     elif a[i]=="Q":
#         b[i]=12
#     elif a[i]==[K]:
#         b[i]=13
#     else:
#         b[i]=int(a[i])

#
# a,b=map(int,input().split())
# print(a+b)
#用selenium库
# from selenium import webdriver
# import time
#路劲之谜
#
# import os
# import sys
#
# sys.setrecursionlimit(60000)
# n = 0
# flag = [[0 for i in range(26)] for i in range(27)]
# resX = [0 for i in range(1000)]
# resY = [0 for i in range(1000)]
#
# resCount = 0
# dx = [0, 1, -1, 0]
# dy = [1, 0, 0, -1]
#
#
# # 下，右，左，上
#
# def check(x, y):
#     global n
#     if x == n and y == n:
#         for i in range(1, n + 1):
#             if (col[i] != 0):
#                 return False
#         for i in range(1, n + 1):
#             if (rol[i] != 0):
#                 return False
#         for i in range(0, resCount):
#             x2 = resX[i]
#             y2 = resY[i]
#             sum = n * (x2 - 1) + y2 - 1
#             print(sum, end="")
#         return False
#
#     else:
#         return True
#
#
# def pd(x2, y2):
#     global n
#     print("x2:", x2)
#     if flag[x2][y2] == 1:
#         return False
#     elif x2 < 1:
#         return False
#
#     elif x2 > n:
#         return False
#     elif col[x2] <= 0:
#         return False
#     elif y2 < 1:
#         return False
#     elif y2 > n:
#         return False
#     elif rol[y2] <= 0:
#         return False
#     else:
#         return True
#
#
# def dfs(x, y):
#     if not check(x, y):
#         return
#     else:
#         for i in range(0, 4):
#             xt = dx[i] + x
#             yt = dy[i] + y
#             if not pd(xt, yt):
#                 continue
#
#             else:
#                 col[xt] -= 1
#                 rol[yt] -= 1
#                 flag[xt][yt] = 1
#                 global resCount
#                 resX[resCount] = xt
#                 resY[resCount] = yt
#                 resCount += 1
#                 dfs(xt, yt)
#                 resCount -= 1
#                 flag[xt][yt] = 0
#                 col[xt] += 1
#                 rol[yt] += 1
#
#
# if __name__ == "__main__":
#     n = int(input())
#     rol = input().split()
#     rol = list(map(int, rol))
#     rol = [0] + rol
#     col = input().split()
#     col = list(map(int, col))
#     col = [0] + col
#
#     flag[1][1] = 1
#     col[1] -= 1
#     rol[1] -= 1
#
#     resX[resCount] = 1
#     resY[resCount] = 1
#     resCount += 1
#     dfs(1, 1)
#

# print(pow(3,20))


#
# cnt=0
# for i in range(50):
#     for j in range(50):
#         for k in range(50):
#             a=3**i
#             b=5**j
#             c=7**k
#             if a*b*c<=590867:
#                 cnt+=1
# print(cnt)
#
#

#
# print(2>>3) # 2//2^3 = 0，2的二进制10，向右最多移动2位后，所以多移动无疑为0
# print(2>>1) # 2*2^1 = 4，向右移动一位为01,
# print(3>>4) # 3*2^4 = 48,3的二进制为11，向右移动四位后00
# print(3>>1) # 3*2^4 = 48,3的二进制为11，向右移动一位后为01
#
#
# a=[1,2,3,4]
# n=3
#
# def print_s(n):
#     for i in range(1<<n):
#         for j in range(n):
#             if (i &(1<<j))!=0:
#                print(a[j],end="")
#         print()
#
# print_s(n)
#
#
# chosen=[]
# n=3
# m=2
#
# def calc(x):
#   if len(chosen)>m:
#     return
#   if len(chosen)+n-x+1<m:
#     return
#   if x==n+1:
#     for i in chosen:
#       print(i,end="")
#     print("")
#     return
#   chosen().append()


from itertools import combinations

# 从用户输入中读取n和m
# n, m = map(int, input().split())
#
# # 输入每个人的名字
# names = [input() for i in range(n)]
#
# # 生成所有可能的组合
# def enumerate_combinations(arr, data, start, end, index, r):
#     # 当组合大小达到m时打印
#     if index == r:
#         print(data)
#         return
#
#     # 递归地从剩余的人中选择一个
#     i = start
#     while i <= end and end - i + 1 >= r - index:
#         data[index] = arr[i]
#         enumerate_combinations(arr, data, i + 1, end, index + 1, r)
#         i += 1
#
# # 创建一个数组来存储所有组合
# data = [0]*m
#
# # 打印所有组合
# enumerate_combinations(names, data, 0, n-1, 0, m)

# from itertools import combinations
#
# # 从用户输入中读取 n 和 m
# n, m = map(int, input().split())
#
# # 读取报名人员的姓名
# names = [input() for i in range(1, n + 1)]
#
# # 生成并打印所有组合
# all_combinations = list(combinations(names, m))
# for i, combo in enumerate(all_combinations, 1):
#     print(f"{' '.join(combo)}")


# from itertools import combinations
#
# # 从用户输入中读取 n 和 m
# n=int(input())
#
# # 读取报名人员的姓名
# names = [input() for i in range(1, n + 1)]
#
# # 生成并打印所有组合
# all_combinations = list(combinations(names, n))
# for i, combo in enumerate(all_combinations, 1):
#     print(f"{' '.join(combo)}")
# from itertools import permutations
#
# # Ask for the number of people
# m = int(input())
#
# # Input the names of the people
# people_names = [input() for i in range(m)]  # Change m+1 to m
#
# # Calculate and print all possible permutations
# for perm in permutations(people_names):
#     print(' '.join(perm))
#
# from itertools import combinations
# # m=int(input())
# n,m=map(int,input().split(" "))
# names=[input() for i in range(1,n+1)]
# # for perm in permutations(names):
# #   print(' '.join(perm))
#
# all=list(combinations(names,m))
# for i,combo in enumerate(all,1):
#   print(f"{' '.join(combo)}")

# a=[0]*100000
# sum1=[0]*10000
# if __name__=="__main__":
#   n,m=map(int,input().split())
#   a=list(map(int,input().split()))
#   for i in range(1,n+1):
#     sum1[i]=sum1[i-1]+a[i-1]
#   while m>0:
#     m-=1
#     l,r=map(int,input().split())
#     print(sum1[r]-sum1[l-1])


a=[0]*100005
b=[0]*100005
if __name__=="__main__":
  n,m=map(int,input().split())
  # a=list(map(int,input().split()))
  for i in range(1,n+1):
    b[i]=a[i]+a[i-1]
  while m>0:
    m-=1
    l,r,v=map(int,input().split())
    l+=1
    r+=1
    b[l]+=v
    b[r+1]-=v
  sum=0
  for i in range(1,n+1):
    a[i]=a[i-1]+b[i]
    sum+=a[i]
  print(sum)