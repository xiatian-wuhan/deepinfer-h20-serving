import sys
p, want = sys.argv[1], sys.argv[2]
s = open(p).read()
old3 = "('NCCL_GIN_TYPE', '3', True)"
new = "('NCCL_GIN_TYPE', '%s', True)" % want
n = s.count(old3)
s = s.replace(old3, new)
open(p, 'w').write(s)
import ast; ast.parse(open(p).read())
print("patch_gin: replaced %d occurrence(s) with type %s; syntax OK" % (n, want))
