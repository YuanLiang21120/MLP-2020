from . import nlp
from . import kmeans


def run(args):
    if args.task.startswith('nlp'):
        nlp.run(args)
    elif args.task.startswith('kmeans'):
        kmeans.run(args)
    else:
        assert false, 'Unknown task'
    print('Done.')
