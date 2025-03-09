# n*n 非负数， 求左上角到右下角最短路径值
#
# 1 2 3 4
# 5 6 7 8
# 1 100 11 12
# 13 14 15 16

a = [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16]
]
n = len(a)
dp = [[0] * n for i in range(n)]
dp[0][0] = a[0][0]

for i in range(1,n):
     dp[0][i] = dp[0][i-1] + a[0][i]
for i in range(1,n):
     dp[i][0] = dp[i-1][0] + a[i][0]

for i in range(1,n):
     for j in range(1,n):
          dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]),min( dp[i+1],[j],dp[i][j+1])) + a[i][j]

print(dp[n-1][n-1])