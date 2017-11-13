import configparser
import cv2
from glob import glob
import configparser
from os.path import join


def __getImgPath__(configFile, parName):
    #  -- read parameters from config --
    config = configparser.ConfigParser()
    config.read(configFile)
    try:
        def configSectionMap(section):
            dict1 = {}
            options = config.options(section)
            for option in options:
                try:
                    dict1[option] = config.get(section, option)
                    if dict1[option] == -1:
                        print("skip: %s" % option)
                except:
                    print("exception on %s!" % option)
                    dict1[option] = None
            return dict1

        if (configSectionMap("default")[parName]) :
            return configSectionMap("default")[parName]
        else:
            return input('Input image dir path: ')

    # -- read parameters from input --
    except KeyError:
        return input('Input image dir path: ')
    except configparser.NoSectionError:
        return input('Input image dir path: ')


def getImages(imageDirPath):
    print('imageDirPath = ' + imageDirPath)

    types = ('*.jpeg', '*.JPG', '*.gif', '*.png', '*.jpg', '*.bmp')
    filesPaths = []

    for extension in types:
        filesPaths.extend(glob(join(imageDirPath, extension)))

    imgs = []
    for path in filesPaths:
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path, 3)
        imgs.append(img)

    # -- show images --
    if (len(imgs) == 0):
        print("Images not found")
        exit()
    return imgs



def getParamFromConfig(parName):
    imageDirPath = __getImgPath__('config.ini', parName)
    return imageDirPath


