# import datetime
# def dayofyear():
#     year = input("请输入年份: ")
#     month = input("请输入月份: ")
#     day = input("请输入天: ")
#     date1 = datetime.date(year=int(year),month=int(month),day=int(day))
#     date2 = datetime.date(year=int(year),month=1,day=1)
#     return (date1-date2).days+1
# print(dayofyear())


# str1 = "k:1|k1:2|k2:3|k3:4"
# def str2dict(str1):
#     dict1 = {}
#     for iterms in str1.split('|'):
#         key,value = iterms.split(':')
#         dict1[key] = value
#     return dict1
# #字典推导式
# d = {k:int(v) for t in str1.split("|") for k, v in (t.split(":"), )}
#
# list = ['a','b','c','d','e']
# print(list[10:])

# print([x*11 for x in range(10)])
# A=[1,2,3,4,5,6,7,8,9]
# B=[1,3,5,7,9,11,13,15]
# print(set(A)^set(B))
#
# import os
#
# def get_files(dir, suffix):
#     res = []
#     for root, dirs, files in os.walk(dir):
#         for filename in files:
#             name, suf = os.path.splitext(filename)
#             print(name, suf)
#             if suf == suffix:
#                 res.append(os.path.join(root, filename))
#     return res
#
# result = get_files("./", '.xml')
# print(result)

# a = [1,2,3,4,5,6,7,8]
# print(id(a))
# print(id(a[:]))
# for i in a[:]:
#     if i>5:
#         pass
#     else:
#         a.remove(i)
#     print(a)
# print('-----------')
# print(id(a))

a=[1,2,3,4,5,6,7,8]
b = filter(lambda x: x>5,a)
print(list(b))
