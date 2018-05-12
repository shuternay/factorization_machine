# Устройство утилиты

При чтении датасета имена фичей заменяются на их хэш (аналогично тому, как это делает vw), хэшфункция принимает `2^b` значений, где `b` задаётся при запуске.

Есть два режима обучения: градиентный спуск (SGD или AdaGrad) и ALS.

При обучении в несколько потоков применяется Hogwild (как в градиентном спуске, так и в ALS).

Возможно чтение датасетов в форматах csv и libsvm.

## Градиентный спуск

Возможна оптимизация трёх loss-функций: mse, mae и logistic

### SGD

Размер шага вычисляется по следующей формуле (взято у vw):
```
λ d^k / t,
```
где `λ` &mdash; learning rate, `d` &mdash; learning rate decay, `k` &mdash; число полных эпох обучения, `t` &mdash; число итераций (просмотренных записей) обучения.

### AdaGrad

Размер шага для параметра `i` вычисляется по формуле
```
λ d^k / t_i,
```
где `t_i = sqrt(sum g_{i,j}^2)`, `g_{i,j}` &mdash; градиент `i`-го параметра на `j`-й итерации обучения.


### ALS

Алгоритм реализован по статье от [libFM](https://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf).

Можно попытаться сделать ALS для произвольной loss-функции, если посчитать вторую производную loss-функции по предсказанию. Тогда формула (22) в статье будет выглядеть так:

![.](https://latex.codecogs.com/gif.latex?%24%24%20%5Ctheta%5E*%20%3D%20%5Cfrac%20%7B%5Ctheta%20%5Csum_%7Bi%3D1%7D%5En%20h_%5Ctheta%5E2%28%5Cvec%20x_i%29%20%5Cmathcal%7BL%7D_%7B%5Chat%20y%7D%27%27%28%5Cvec%20x_i%29%20-%20%5Csum_%7Bi%3D1%7D%5En%20h_%5Ctheta%28%5Cvec%20x_i%29%20%5Cmathcal%7BL%7D_%7B%5Chat%20y%7D%27%28%5Cvec%20x_i%29%7D%20%7B%5Csum_%7Bi%3D1%7D%5En%20h_%5Ctheta%5E2%28%5Cvec%20x%29%20%5Cmathcal%7BL%7D_%7B%5Chat%20y%7D%27%27%28%5Cvec%20x_i%29%20&plus;%20%5Clambda_%5Ctheta%7D%20%24%24)

Эта штука взрывается, если вторая производная маленькая (mae или logistic при предсказаниях близких к правильному), но если её ограничить снизу константой вроде 0.1~1.0, то вроде работает. На нормальные эксперименты времени не хватило.

# Установка

Утилита написана на Rust.

Сборка утилиты:
```
cargo build --release
```

# Запуск

Команда запуска
```
cargo run -q --release -- <utility options>
```

Опции:
```
-h,--help                   Справка по опциям утилиты
-p,--predict                Запуск в режиме предсказания. По умолчанию утилита запускается в режиме обучения
-d,--data DATA              Путь к датасету для обучения или вычисления предсказаний
--data_type DATA_TYPE       Тип датасета. Возможные значения: csv, libsvm. Значение по умолчанию: libsvm
-t,--target TARGET          Поле таргета для csv датасетов
-m,--model MODEL            Путь к модели, куда записывается обученная модель или откуда берётся модель для предсказания
-o,--output OUTPUT          Путь к файлу для записи вычисленных предсказаний. При отсутствии предсказания не выводятся.
--opt OPT                   Тип оптимизатора. Возможные значения: sgd, adagrad, als. Значение по умолчанию: adagrad.
--loss LOSS                 Лосс функция. Возможные значения: mse, logistic, mae. Значение по умолчанию: mse
-i,--iterations ITERATIONS  Число эпох обучения. Значение по умолчанию: 10
-b,--bits BITS              Число бит хэш-функции. Значение по умолчанию: 18
-k,--factors_number         Число факторов в модели: Значение по умолчанию: 10
--l2 L2                     Значение l2-регуляризатора. Значение по умолчанию: 1e-5
--lr LR                     Значение learning rate для градиентного спуска
--decay DECAY               Значение learning rate decay для градиентного спуска
-j,--jobs JOBS              Число потоков
```

## Примеры:

Обучение с AdaGrad:
```
cargo run -q --release -- -d datasets/train_20m_wo_time.csv --data_type csv --target rating -m model --loss mse -i 20 -j 8
```

Обучение с ALS и вычислением скора на тесте после каждой итерации:
```
cargo run -q --release -- -d datasets/train_20m_wo_time.csv --test_data datasets/test_20m_wo_time.csv --data_type csv --target rating -m model --loss mse -i 20 -j 8 --opt als
```

Вычисление скора на тесте:
```
cargo run -q --release -- -p -d datasets/test_20m_wo_time.csv --data_type csv --target rating -m model --loss mae
```


# Бенчмарки

## Movielens

Для vowpal wabbit использовался [этот](https://github.com/JohnLangford/vowpal_wabbit/tree/master/demo/movielens) бенчмарк с заменой датасета на 20m.
Результаты:
```
linear test MAE is 0.652
lrq test MAE is 0.639
lrqdropout test MAE is 0.608
lrqdropouthogwild test MAE is 0.787
```

### Наша утилита:

AdaGrad, 8 факторов:
```
train mse: 0.592
test mse: 0.677
train mae: 0.586
test mae: 0.627
```
Минимум на тесте около 10-й итерации.

Минимум ошибки на тесте достигается при ~8 факторах.


ALS, 8 факторов, l2 1e-6 (~15), 20 итераций
```
train mse: 0.593
test mse: 0.672
train mae: 0.588
test mae: 0.625
```

ALS, 12 факторов, l2 1e-6 (~15), 20 итераций
```
train mse: 0.566
test mse: 0.669
train mae: 0.573
test mae: 0.622
```


### libFM:

SGD, 8 факторов, lr 0.01, 17 итераций
```
train mse: 0.756
test mse: 0.811
```

SGD, 8 факторов, lr 0.01, 60 итераций
```
train mse: 0.749
test mse: 0.807
```

ALS, 8 факторов, l2 10, 40 итераций
```
train mse: 0.758
test mse: 0.804
```

ALS, 8 факторов, l2 10, 40 итераций
```
train mse: 0.736
test mse: 0.799
```

MCMC, 8 факторов, 70 итераций
```
train mse: 0.768
test mse: 0.798
```


## Время работы

### Movielens

Загрузка train и test датасетов, 8 факторов:
```
our fm: 10 с.
libFM: 36 с.
```

Загрузка датасетов + 10 итераций AdaGrad / SGD, 8 факторов:
```
our fm: 164 с.
our fm, 4 потока: 65 c. (206 с. проц. время)
our fm, 8 потоков: 50 с. (268 с. проц. время)
libFM: 121 c. 
```


Загрузка датасетов + 10 итераций ALS, 8 факторов:
```
our fm: 295 с. 
our fm, 4 потока: 112 c. (374 с. проц. время)
our fm, 8 потоков: 105 с. (661 с. проц. время)
libFM: 295 с.
```

