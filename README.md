# Intro_to_the_Math_of_intelligence

[![youtube_link](https://img.youtube.com/vi/xRJCOz3AfYY/0.jpg)](https://youtu.be/xRJCOz3AfYY)

Porting of the [code](https://github.com/llSourcell/Intro_to_the_Math_of_intelligence) for "Intro - The Math of Intelligence" by Siraj Raval on Youtube to make it runnable on [FloydHub](https://www.floydhub.com/).

## Overview

This is the code for [this](https://youtu.be/xRJCOz3AfYY) video on Youtube by Siraj Raval. The dataset represents distance cycled vs calories burned. We'll create the line of best fit (linear regression) via gradient descent to predict the mapping. yes, I left out talking about the learning rate in the video, we're not ready to talk about that yet.

Here are some helpful links:


#### Gradient descent visualization

![gsd_example](https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif)

> https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif


#### Sum of squared distances formula (to calculate our error)

![SQE](https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png)

> https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png


#### Partial derivative with respect to b and m (to perform gradient descent)

![partial derivative](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)

> https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png


## FloydHub Setup

Here's the commands to launch the demo on FloydHub.


### 1. Create a FloydHub account

- [Sign up](https://www.floydhub.com/signup) on FloydHub
- Install the floyd CLI on your local machine through these two [steps](https://www.floydhub.com/welcome):

```bash
$ pip install -U floyd-cli

$ floyd login
# Follow the instructions on your CLI
```

### 2. Get this project to your local machine

Clone from FloydHub:

```bash
$ cd /path/to/your-project-dir
$ floyd clone llSourcell/projects/Intro_to_the_Math_of_intelligence/1
```

Clone from Github:

```bash
$ git clone https://github.com/floydhub/Intro_to_the_Math_of_intelligence
$ cd Intro_to_the_Math_of_intelligence
```

### 3. Create your project version on FloydHub

[Create a project](https://www.floydhub.com/projects/create) on FloydHub and then sync the cloned repository with your new project

```bash
$ floyd init Intro_to_the_Math_of_intelligence
```

### 4. Run on FloydHub

The `--data` flag specifies that the version 1 of the distance-vs-calories dataset should be available at the `/datasets` directory. *Note*: If you want to mount/create a dataset look at the [docs](http://docs.floydhub.com/guides/basics/create_new/#create-a-new-dataset) and our last [blog post](https://blog.floydhub.com/creating-datasets-from-public-urls/).
The `--env` flag specifies the environment that this project should run on, which is a Tensorflow 1.1.0 + Keras 2.0.6 backend environment with Python 3.5. Even if this is a basic tutorial that can run on every [FloydHub environments](https://docs.floydhub.com/guides/environments/), we suggest you to always specify an environment, this minimize all the reproducibility issue.

```bash
floyd run \
  --env tensorflow-1.0 \
  --data llSourcell/datasets/distance-vs-calories/1:dataset \
  "python demo.py"
```

You can follow along the progress by using the [logs](https://docs.floydhub.com/commands/logs/) command or looking at the logs Panel inside the Overview Tab of your web dashboard's Job. This is the output of the demo:

```bash
...
################################################################################

2017-11-07 03:11:37,795 INFO - Run Output:
2017-11-07 03:11:37,992 INFO - Starting gradient descent at b = 0, m = 0, error = 5565.107834483211
2017-11-07 03:11:37,992 INFO - Running...
2017-11-07 03:11:38,124 INFO - After 1000 iterations b = 0.08893651993741346, m = 1.4777440851894448,
error = 112.61481011613473
2017-11-07 03:11:38,185 INFO -
################################################################################
...
```


## More about FloydHub platform

- The Output of your Job is returned *only if it saved inside the `/output` folder*, see our [docs](https://docs.floydhub.com/guides/data/storing_output/) for a more detailed explanation.

- A keypoint of your experiments and a data science best pratice is to have a clean separation of the code from the data that it uses. This will allow you to structure the experiments/Jobs in a more elegant way and optimize the code you need to upload on FloydHub and speed up the experiment cycle iterations.

- If you need any help check our [documentation](http://docs.floydhub.com/) and [forum](https://forum.floydhub.com/).

## Coding Challenge

This week's coding challenge is to implement gradient descent to find the line of best fit that predicts the relationship between 2 variables of your choice from a [FloydHub](https://www.floydhub.com/datasets) or [kaggle](https://www.kaggle.com/datasets) dataset. *Bonus points for detailed documentation*. Good luck!

We encourage you to try this alone and then check the winner of the coding challenge(wizard of the week) in the next Lesson.

## Credits

Credits for this code go to [mattnedrich](https://github.com/mattnedrich). I've merely created a wrapper to get people started.
