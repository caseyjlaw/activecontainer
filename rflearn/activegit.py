from sh import git

# assume we have
# - training.pkl -- dictionary with training set keys of features and values of 1/0
# - testing.pkl -- dictionary with testing set keys of features and values of 1/0
# - classifier.pkl -- classifier for latest training set
#
# git tag is set to rbversion

class ActiveGit():
    """ 
    """

    def __init__(self, repopath):

        self.repo = git.bake(_cwd=repopath)

        try:
            print('Testing repo status')
            print(self.repo.status())
        except OSError:
            print('No repo found. Making new one...')
            os.mkdir(repopath)
            self.repo.init()


    def listversions(self):
        print('Tags available:')
        stdout = self.repo.tag())
        print(stdout)


    def latestversion(self):
        stdout = self.repo.checkout('master')
        print(stdout)


    def getversion(self, version):
        stdout = self.repo.checkout(version)
        print(stdout)


    def read_training_data(self):
        """ Read data dictionary from training.pkl """

        data = pickle.load(open('training.pkl'))
        return data.keys, data.values


    def read_testing_data(self):
        """ Read data dictionary from testing.pkl """

        data = pickle.load(open('testing.pkl'))
        return data.keys, data.values


    def write_training_data(self, data):
        """ Write data dictionary to filename """

        with open('training.pkl', 'w') as fp:
            pickle.dump(data, fp)


    def write_testing_data(self, data):
        """ Write data dictionary to filename """

        with open('testing.pkl', 'w') as fp:
            pickle.dump(data, fp)


    def updateversion(self, version, msg=''):
        """ Add tag, commit, and push changes """

        self.repo.tag(version)
        self.repo.commit(m=msg)
        self.repo.push('origin', '--tags') # be sure to push tags every time an update is made
