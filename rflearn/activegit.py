from sh import git
import os, pickle


std_files = ['classifier.pkl', 'testing.pkl', 'training.pkl']


class ActiveGit():
    """ Use tags to identify versions of an active learning loop and classifier.

    Assumes that repo has 'training.pkl', 'testing.pkl', and 'classifier.pkl'.
    First two contain single dictionary with features as keys and target (0/1) as values.
    The third object is the sklearn random forest object.

    A version can be any string (such as used in rbversion), but can also be based on name of expert doing classification.

    Tags are central to tracking classifier and data. 
    Branch 'master' keeps latest and branch 'working' is used for active session.
    """

    def __init__(self, repopath):
        """ Set up activegit for repo at repopath. Checks out master branch (lastest commit) by default. """

        self.repo = git.bake(_cwd=repopath)
        self.repopath = repopath

        if os.path.exists(repopath):
            try:
                contents = [gf.rstrip('\n') for gf in self.repo.bake('ls-files')()]
                if all([sf in contents for sf in std_files]):
                    print('ActiveGit initializing from repo at {0}'.format(repopath))
                    print('Available versions: {0}'.format(','.join(self.versions)))
                    if 'working' in self.repo.branch().stdout:
                        print('Found working branch on initialization. Removing...')
                        cmd = self.repo.checkout('master')
                        cmd = self.repo.branch('working', d=True)
                    self.set_version(self.repo.describe(abbrev=0, tags=True).stdout.rstrip('\n'))
                else:
                    print('{0} does not include standard set of files {1}'.format(repopath, std_files))
            except:
                contents = os.listdir(repopath)
                if all([sf in contents for sf in std_files]):
                    print('Uninitialized repo found at {0}. Initializing...'.format(repopath))
                    cmd = self.repo.init()
                    cmd = self.repo.add('training.pkl')
                    cmd = self.repo.add('testing.pkl')
                    cmd = self.repo.add('classifier.pkl')
                    cmd = self.repo.commit(m='initial commit')
                    cmdn = self.repo.tag('initial')
                    cmd = self.set_version('initial')
                else:
                    print('{0} does not include standard set of files {1}'.format(repopath, std_files))
        else:
            print('No repo or directory found at {0}'.format(repopath))


    # version/tag management

    @property
    def version(self):
        if hasattr(self, '_version'):
            return self._version
        else:
            print('No version defined yet.')


    @property
    def versions(self):
        return sorted(self.repo.tag().stdout.rstrip('\n').split('\n'))


    @property
    def isvalid(self):
        gcontents = [gf.rstrip('\n') for gf in self.repo.bake('ls-files')()]
        fcontents = os.listdir(self.repopath)
        return all([sf in gcontents for sf in std_files]) and all([sf in fcontents for sf in std_files])
        

    def set_version(self, version, force=False):
        if version in self.versions:
            self._version = version
            if 'working' in self.repo.branch().stdout:
                if force:
                    print('Found working branch. Removing...')
                    cmd = self.repo.checkout('master')
                    cmd = self.repo.branch('working', d=True)                    
                else:
                    print('Found working branch from previous session. Use force=True to remove it and start anew.')
                    return

            stdout = self.repo.checkout(version, b='working').stdout  # active version set in 'working' branch
            print('Version {0} set'.format(version))
        else:
            print('Version {0} not found'.format(version))


    def show_version_info(self, version):
        if version in self.versions:
            stdout = self.repo.show(version, '--summary').stdout
            print(stdout)
        else:
            print('Version {0} not found'.format(version))


    # data read/write methods

    def read_training_data(self):
        """ Read data dictionary from training.pkl """

        data = pickle.load(open(os.path.join(self.repopath, 'training.pkl')))
        return data.keys(), data.values()


    def read_testing_data(self):
        """ Read data dictionary from testing.pkl """

        data = pickle.load(open(os.path.join(self.repopath, 'testing.pkl')))
        return data.keys(), data.values()

    def read_classifier(self):
        """ Load classifier from classifier.pkl """        

        clf = pickle.load(open(os.path.join(self.repopath, 'classifier.pkl')))
        return clf


    def write_training_data(self, features, targets):
        """ Write data dictionary to filename """

        assert len(features) == len(targets)

        data = dict(zip(features, targets))

        with open(os.path.join(self.repopath, 'training.pkl'), 'w') as fp:
            pickle.dump(data, fp)


    def write_testing_data(self, features, targets):
        """ Write data dictionary to filename """

        assert len(features) == len(targets)

        data = dict(zip(features, targets))

        with open(os.path.join(self.repopath, 'testing.pkl'), 'w') as fp:
            pickle.dump(data, fp)

    def write_classifier(self, clf):
        """ Write classifier object to pickle file """

        with open(os.path.join(self.repopath, 'classifier.pkl'), 'w') as fp:
            pickle.dump(clf, fp)


    # methods to commit, pull, push

    def commit_version(self, version, msg=None):
        """ Add tag, commit, and push changes """

        assert version not in self.versions, 'Will not overwrite a version name.'

        if not msg:
            feat, targ = self.read_training_data()
            msg = 'Training set has {0} examples. '.format(len(feat))
            feat, targ = self.read_testing_data()
            msg += 'Testing set has {0} examples.'.format(len(feat))

        cmd = self.repo.commit(m=msg, a=True)
        cmd = self.repo.checkout('master')
        self.update()
        cmd = self.repo.merge('working')
        cmd = self.repo.branch('working', d=True)
        cmd = self.repo.tag(version)

        try:
            stdout = self.repo.push('origin', 'master', '--tags').stdout
            print(stdout)
        except:
            print('Push not working. Remote not defined?')


    def update(self):
        """ Pull latest versions/tags, if linked to github. """

        try:
            stdout = self.repo.pull().stdout
            print(stdout)
        except:
            print('Pull not working. Remote not defined?')
