# Declaring features:
"""
Here are lists containing features for doing SVM and Decision tree algorithm,
each one is a different iteration. The list contains two elements, the first
of wich is the list containing feature_name in the csv file, while the second
of which is the list of its corresponding translation in Chinese for reading
enhancement. Examples are:
key_1 = [[feature_names], [corresponding_translations]]
"""
test_key = [['time', 'slowCool', 'pH', 'leak', 'numberInorg', 'numberOrg',
            'numberOxlike', 'numberComponents', 'orgavgpolMax',
            'orgrefractivityMax', 'orgmaximalprojectionareaMax',
            'orgmaximalprojectionradiusMax', 'orgmaximalprojectionsizeMax',
            'orgminimalprojectionareaMax', 'orgminimalprojectionradiusMax',
            'orgminimalprojectionsizeMax', 'orgavgpol_pHdependentMax',
            'orgmolpolMax', 'orgvanderwaalsMax', 'orgASAMax', 'orgASA+Max',
            'orgASA-Max', 'orgASA_HMax', 'orgASA_PMax', 'orgpolarsurfaceareaMax',
            'orghbdamsaccMax', 'orghbdamsdonMax', 'orgavgpolMin', 'orgrefractivityMin',
            'orgmaximalprojectionareaMin', 'orgmaximalprojectionradiusMin',
            'orgmaximalprojectionsizeMin', 'orgminimalprojectionareaMin'],
            ['反应时间', '慢速冷却', 'pH', '产物泄露', '无机组分数', '有机组分数量',
             'Ox类似物数量', '组分数', '有机物平均分子极化率Max',
            '有机物最大折射率', '有机物maximalprojectionareaMax',
            '有机物maximalprojectionradiusMax', '有机物maximalprojectionsizeMax',
            '有机物minimalprojectionareaMax', '有机物minimalprojectionradiusMax',
            '有机物minimalprojectionsizeMax', '有机物平均分子极化率-pH依赖关系Max',
            '有机物最大分子极性', '有机物最大范德华立体表面积', '有机分子最大亲水表面积', '有机物最大亲水表面积+',
            '有机物最大亲水表面积', '有机物最大亲水表面积_H', '有机物最大亲水表面积_P', '有机物最大极性表面积',
            '有机物最大氢键受体数量', '有机物最大氢键给体数', '有机物平均分子极化率Min', '有机物最小折射率',
            '有机物maximalprojectionareaMin', '有机物maximalprojectionradiusMin',
            '有机物maximalprojectionsizeMin', '有机物minimalprojectionareaMin'
             ]
            ]
feat_first = [['orgvanderwaalsMin', 'orgASA+Min', 'orghbdamsdonGeomAvg',
              'PaulingElectronegMean', 'hardnessMaxWeighted', 'AtomicRadiusMeanWeighted'],
              ['orgMin', '有机物最小亲水表面积+', '有机物氢键给体几何平均值',
               '平均Pauling电负性', '最大质量的硬度', '重均原子半径']
              ]

feat_top6 = [['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg'],
             ['反应时间', '最低质量的硬度', '有机物亲水表面积_H几何平均', '产物泄露',
             '无机组分/水摩尔比', '有机物范德华立体表面积算术平均']
             ]

feat_top9 = [['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg',
             'orghbdamsaccMax', 'temp', 'EAMinWeighted'],
             ['反应时间', '最低质量的硬度', '有机物亲水表面积_H几何平均', '产物泄露',
             '无机组分/水摩尔比', '有机物范德华立体表面积算术平均',
             '有机物最大氢键受体数量', '反应温度', '最低质量的电子亲和能']
             ]
