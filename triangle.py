import random
import sys
import cv2
import numpy as np
from PIL import Image

x = 1200
y = 960
# tri_num = 50
tri_num = 100
length = tri_num * 7
population = 100
gap = 0.9
elite_rate = 1.0
p_mutate = 0.05
p_cross = 1.0
# generation = 10
generation = 15
tournament_size = 5
gen = 0
child_num = 0
picture_name = 'himejijo.jpg'

max_list = []
min_list = []
avg_list = []

class ga_individual:
    # gtype = []
    def __init__(self, fitness, gtype):
        self.fitness = fitness
        self.gtype = gtype
    
    def mk_random_gtype(self):
        for i in range(tri_num):
            for j in range(3):
                self.gtype.append(random.randint(-100, x+100))
                # print(self.gtype)
                self.gtype.append(random.randint(-100, y+100))
            self.gtype.append(random.randint(0, 255))
    
    def draw_triangles(self, child_num):
        img = []
        pts = []
        color = []
        for i in range(tri_num):
            img.append(np.full((y, x, 4), 255, dtype=np.uint8))
            pts.append(np.array(((self.gtype[i*7], self.gtype[i*7+1]), (self.gtype[i*7+2], self.gtype[i*7+3]), (self.gtype[i*7+4], self.gtype[i*7+5]))))
            # print(img[i])
            cv2.fillPoly(img[i], [pts[i]], (self.gtype[i*7+6], self.gtype[i*7+6], self.gtype[i*7+6], 128))
            # print("a")
        # out_img = (img[0]/10+img[1]/10+img[2]/10+img[3]/10+img[4]/10+img[5]/10+img[6]/10+img[7]/10+img[8]/10+img[9]/10)
            img[i][:, :, 3] = np.where(np.all(img == 255, axis=-1), 0, 255) 
        out_img = img[0]/tri_num*4
        for i in range(1, tri_num):
            out_img += img[i]/tri_num*4

        Image.fromarray(out_img.astype(np.uint8)).save(f'himejijo_image/out{gen}_{child_num}.png')
            # out_img = cv2.bitwise_and(out_img, img[i])
            # out_img = cv2.addWeighted(out_img, 1.0-1.0/tri_num, img[i], 1.0/tri_num, 0)
        # img = np.full((540, 800, 3), 255, dtype=np.uint8)
        # for i in range(tri_num):
        #     pts = np.array(((self.gtype[i*7], self.gtype[i*7+1]), (self.gtype[i*7+2], self.gtype[i*7+3]), (self.gtype[i*7+4], self.gtype[i*7+5])))
        #     cv2.fillPoly(img, [pts], (self.gtype[i*7+6], self.gtype[i*7+6], self.gtype[i*7+6]))
        # cv2.imwrite(f'out_image/out{gen}_{child_num}.png', out_img)   

    def calc_mse(self, child_num):
        picture = cv2.imread(picture_name, cv2.IMREAD_GRAYSCALE)
        out_image = cv2.imread(f'himejijo_image/out{gen}_{child_num}.png', cv2.IMREAD_GRAYSCALE)
        mse = 0.0
        for i in range(y):
            for j in range(x):
                mse += abs(int(picture[i][j]) - int(out_image[i][j]))
                # mse += abs(int(picture[i][j]))
        mse /= x*y
        return mse
     
            
            
class ga_population:
    def __init__(self, genes, pselect, mutate_count, max_fitness, min_fitness, avg_fitness):
        self.genes = genes
        self.pselect = pselect
        self.mutate_count = mutate_count
        self.max_fitness = max_fitness
        self.min_fitness = min_fitness
        self.avg_fitness = avg_fitness

    def select_parent_tournament(self):
        min = population-1
        for i in range(tournament_size):
            r = random.randint(0, population-1)
            if min > r:
                min = r
        min_selected = self.genes[min]
        # print(min)
        return min_selected

    #適合度順に並んだ線形リストから最大値、最小値、平均値を記録、順番付け
    def normalize_population(self):
        self.max_fitness = self.genes[0].fitness
        avg = 0.0
        for i in range(population):
            avg += self.genes[i].fitness
            self.min_fitness = self.genes[i].fitness
        avg = avg / population
        self.avg_fitness = avg
    
    def calc_fitness(self):
        for i in range(population):
            child_num = i
            self.genes[i].draw_triangles(child_num)
            mse = self.genes[i].calc_mse(child_num)
            self.genes[i].fitness = 1/(1+abs(mse))
        self.genes.sort(key=lambda u: u.fitness, reverse=True)

def cross_gtype(parent1, parent2):
    cross_point = random.randint(0, length-1)
    child1 = ga_individual(0, [])
    child2 = ga_individual(0, [])
    i = 0
    while i<=cross_point:
        child1.gtype.append(parent1.gtype[i])
        child2.gtype.append(parent2.gtype[i])
        # print(child1.gtype)
        i+=1
    while i<=length-1:
        child1.gtype.append(parent2.gtype[i])
        child2.gtype.append(parent1.gtype[i])
        i+=1

    return child1, child2

def mutate_gtype(gtype):
    if p_mutate>1 or p_mutate < 0.0:
        print(f'{p_mutate} is used for mutation probability, but this must be from 0.0 to 1.0')
        sys.exit()
    mutate_point = 0
    for i in range(length):
        rm = random.random()
        if rm < p_mutate:
            if i % 7 == 6:
                gtype[i] = random.randint(0, 255)
            elif (i % 7) % 2 == 0:
                gtype[i] = random.randint(-100, x+100)
            else:
                gtype[i] = random.randint(-100, y+100)
            mutate_point += 1
    return mutate_point 

def mk_gene():
    gene = ga_individual(0, [])
    gene.mk_random_gtype()
    return gene

def mk_children_genes(parent1, parent2):
    child1, child2 = cross_gtype(parent1, parent2)
    mutate_count = mutate_gtype(child1.gtype)
    mutate_count += mutate_gtype(child2.gtype)
    return mutate_count, child1, child2

def mk_init_ga_population():
    people = ga_population([], [], 0, 0, 0, 0)
    for i in range(population):
        people.genes.append(mk_gene())
        # print(people.genes[i].gtype[:3])
    for j in range(population):
        people.pselect.append(0.0)
    return people

def print_population(people):
    i = 0
    member = people.genes
    print("-------------------"+"-"*(length+2)+"---------------")
    print("#   parents  xsite  gtype"+' '*(length-3)+" fitness")
    for i in range(population):
        print(f'{i}  {member[i].gtype[:10]} {member[i].fitness}')
    print(f'total mutate {people.mutate_count}')

def print_fitness(people):
     print(f'{people.max_fitness}, {people.avg_fitness}, {people.min_fitness} {people.genes[0].gtype[:10]}')
     max_list.append(people.max_fitness)
     min_list.append(people.min_fitness)
     avg_list.append(people.avg_fitness)

# GA集団の個体線形リストgenesの一人一人のfitnessを見て
#    配列pselectを作る
#    1. pselect[i] = pselect[i-1]+fitness[i]
#    2. pselect[i] = pselect[i]/pselect[POPULATION-1]
def calc_pselect(people):
    people.pselect[0] = people.genes[0].fitness
    for i in range(1, population):
        people.pselect[i] = people.pselect[i-1] + people.genes[i].fitness
    for j in range(population):
        people.pselect[i] /= people.pselect[population-1]

# 新しい世代の生成 必ずソート済みのpopulationを渡す
def generate_population(new_people, old_people):
    num_of_remain = (int)(population*(1-gap)) #親世代からコピーする数
    num_of_elite = (int)(num_of_remain*elite_rate) #コピー枠のうちエリートの数
    old_genes = old_people.genes
    new_genes = new_people.genes
    #親選択テーブルを準備
    calc_pselect(old_people)
    #エリート戦略 親世代での上位一定数はそのまま子供になる
    for generated in range(num_of_elite):
        new_genes[generated].gtype = old_genes[generated].gtype
  
    new_people.mutate_count = 0
    #交叉・突然変異を適応する枠
    #残り個体数が奇数の時は、一つだけ突然変異で子供を作る
    if (population - generated)%2 == 1:
        new_genes[generated] = old_people.select_parent_tournament()
        new_people.mutate_count += mutate_gtype(new_genes[generated].gtype)
        generated+=1
    #交叉・突然変異をする
    for i in range(generated, population, 2):
        rand_double = random.random()
        #交叉するとき
        if rand_double < p_cross:
            temp_mc, new_genes[i], new_genes[i+1] = mk_children_genes(old_people.select_parent_tournament(), old_people.select_parent_tournament())
            new_people.mutate_count += temp_mc
        else:
            new_genes[i] = old_people.select_parent_tournament()
            new_people.mutate_count += mutate_gtype(new_genes[i].gtype)
            new_genes[i+1] = old_people.select_parent_tournament()
            new_people.mutate_count += mutate_gtype(new_genes[i+1].gtype)

#main
parent_group = mk_init_ga_population()
child_group = mk_init_ga_population()
print("#generation,max_fitness, avg_fitness, min_fitness, best_individual_gtype")
for gen in range(generation):
    #集団の適合度を計算し、線形リストを作る
    parent_group.calc_fitness()
    #最大値・最小値、
    parent_group.normalize_population()
    #現在の世代の表示
    # print_population(parent_group)
    print(f'{gen} ')
    print_fitness(parent_group)
    #現在の世代parent_groupから次世代child_groupを作る。
    generate_population(child_group,parent_group)
    parent_group = child_group
print("max_list", max_list)
print("min_list", min_list)
print("avg_list", avg_list)