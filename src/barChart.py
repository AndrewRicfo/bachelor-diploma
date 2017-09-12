import numpy as np
import matplotlib.pyplot as plt


# n_groups = 5
# means_men = (20, 35, 30, 35, 27)
# std_men = (2, 3, 4, 1, 2)
# means_women = (25, 32, 34, 20, 25)
# std_women = (3, 5, 2, 3, 3)
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 0.4
# error_config = {'ecolor': '0.3'}
#
# rects1 = plt.bar(index, means_men, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  yerr=std_men,
#                  error_kw=error_config,
#                  label='Men')
#
# rects2 = plt.bar(index + bar_width, means_women, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  yerr=std_women,
#                  error_kw=error_config,
#                  label='Women')
#
# rects3 = plt.bar(index + bar_width, means_women, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  yerr=std_women,
#                  error_kw=error_config,
#                  label='Women')
# plt.xlabel('Group')
# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# labels = ('DecisionTrees', 'SVM', 'NN', 'RandomForrests', 'Linear', 'Polynomial(2 degree)', 'Polynomial(3 degree)', 'RBF regression', 'SBF regression')
# # plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
# plt.xticks(index + bar_width / 2, labels, rotation='vertical')
# plt.legend()
# plt.tight_layout()
# plt.show()


plt.rcParams.update({'font.size': 13}) #default = 10
fig, ax = plt.subplots()

bar_width = 0.25
objects = ('DecisionTrees', 'SVM', 'NN', 'RandomForrests', 'Linear regression', 'Polynomial(2 degree)', 'Polynomial(3 degree)', 'RBF regression', 'SBF regression')
y_pos = np.arange(len(objects))

mse = [0.00602687154716,
       0.0219873154884,
       0.004135,
       0.00616560979014,
       0.00430226080824,
       0.0192249918826,
       0.0159199050103,
       0.00311348813515,
       0.00603318397948
       ]

mae = [0.0413641673495,
       0.091002349763,
       0.0396,
       0.0517593954295,
       0.0455439937631,
       0.090228406106,
       0.0718865563663,
       0.0445481752554,
       0.0446336585525]

meder = [0.022999999553,
         0.099897340179,
         0.0227522,
         0.0278494433601,
         0.0289357218811,
         0.0562716308236,
         0.03427781865,
         0.0260697570266,
         0.0338575138671]

plt.bar(y_pos+1, mse, bar_width, align='center', alpha=0.5, color='b')
plt.bar(y_pos+bar_width+1, mae, bar_width, align='center', alpha=0.5, color='r')
plt.bar(y_pos+2*bar_width+1, meder, bar_width, align='center', alpha=0.5, color='g')
plt.xticks(y_pos+1+bar_width, objects)
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
plt.ylabel('Error value')
plt.title('Comparison of existing methods')
plt.legend(['Mean Squared Error', 'Mean Absolute Error', 'Median absolute error'], loc=1)

plt.show()
