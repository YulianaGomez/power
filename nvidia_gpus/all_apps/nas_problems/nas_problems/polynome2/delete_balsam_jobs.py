import sys
from balsam.launcher.dag import BalsamJob

BalsamJob.objects.filter(name__contains=sys.argv[2], workflow=sys.argv[1]).delete()
