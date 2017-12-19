
import numpy as np

# project_dir = iml.getParamFromConfig('projectdir')
# img_path = project_dir + '/resources/paint_test_3/template.png'
#
# img = cv2.imread(img_path, 3)
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # преобразуем в оттенки серого
# _, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# print('len(contours) = {0}'.format(len(contours)))
#
# moment = {'mu21': 0.0, 'nu20': 0.0, 'm30': 0.0, 'nu11': 0.0, 'm02': 0.0, 'nu03': 0.0, 'm20': 0.0,
#         'm11': 0.0, 'mu02': 0.0, 'mu20': 0.0, 'nu21': 0.0, 'nu12': 0.0 - 18, 'nu30': 0.0, 'm10': 0.0,
#         'm03': 0.0, 'mu11': 0.0, 'mu03': 0.0, 'mu12': 0.0, 'm01': 0.0, 'mu30': 0.0, 'm12': 0.0,
#         'm00': 0.0, 'm21': 0.0, 'nu02': 0.0}
#
# for cnt in contours:
#     m = cv2.moments(cnt)
#     for key in moment.keys():
#         moment[key] = moment[key] + m[key]


arr = [10,10,9,10,10, 7, -1]
min_idx = np.argmin(arr)

print('min_idx = {0}'.format(min_idx))
