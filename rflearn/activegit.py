from sh import git
import os, pickle


class ActiveGit():
    """ Use tags to identify versions of an active learning loop and classifier.

    Assumes that repo has 'training.pkl', 'testing.pkl', and 'classifier.pkl'.
    First two contain single dictionary with features as keys and target (0/1) as values.
    The third object is the sklearn random forest object.

    A version can be any string (such as used in rbversion), but can also be based on name of expert doing classification.

    Note that only tags are used, so repo has no branches.
    """

    def __init__(self, repopath):

        self.repo = git.bake(_cwd=repopath)

        if os.path.exists(repopath):
            try:
                contents = self.repo.bake('ls-files')()
                if contents:
                    print('ActiveGit defined from repo at {0}'.format(repopath))
                    print('Available versions: {0}'.format(','.join(self.versions)))
            except:
                print('Uninitialized repo found at {0}. Initializing...'.format(repopath))
                self.repo.init()
                self.repo.add('training.pkl')
                self.repo.add('testing.pkl')
                self.repo.add('classifier.pkl')
                self.repo.commit(m='initial commit')
                self.repo.tag('initial')
                self.set_version('initial')
                self.repo.branch('-d', 'master')
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


    def set_version(self, version):
        if version in self.versions:
            self._version = version
            stdout = self.repo.checkout(version).stdout
            print(stdout)
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

        data = pickle.load(open('training.pkl'))
        return data.keys(), data.values()


    def read_testing_data(self):
        """ Read data dictionary from testing.pkl """

        data = pickle.load(open('testing.pkl'))
        return data.keys(), data.values()


    def write_training_data(self, features, targets):
        """ Write data dictionary to filename """

        assert len(features) == len(targets)

        data = dict(zip(features, targets))

        with open('training.pkl', 'w') as fp:
            pickle.dump(data, fp)


    def write_testing_data(self, features, targets):
        """ Write data dictionary to filename """

        assert len(features) == len(targets)

        data = dict(zip(features, targets))

        with open('testing.pkl', 'w') as fp:
            pickle.dump(data, fp)


    # methods to commit, pull, push

    def commit_version(self, version, msg=''):
        """ Add tag, commit, and push changes """

        if not msg:
            feat, targ = self.read_training_data()
            msg = 'Training set has {0} examples. '.format(len(feat))
            feat, targ = self.read_testing_data()
            msg += 'Testing set has {0} examples.'.format(len(feat))

        self.repo.commit(m=msg, a=True)
        self.repo.tag(version)

        try:
            self.repo.push('origin', '--tags') # be sure to push tags every time an update is made
        except:
            print('Push not working. Remote not defined?')


    def update(self):
        """ Pull latest versions/tags, if linked to github. """

        try:
            self.repo.pull()
        except:
            print('Pull not working. Remote not defined?')
