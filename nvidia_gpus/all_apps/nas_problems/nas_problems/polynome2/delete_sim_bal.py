import sys
from balsam.launcher.dag import BalsamJob

BalsamJob.objects.filter( workflow=sys.argv[1]).delete()
