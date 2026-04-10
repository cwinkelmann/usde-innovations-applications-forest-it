A template git repository for working with Jupyter notebooks.


# Setting up

Different options are available to start working. The most straightforward
option is to access the notebooks via binder (see [section Binder](#Binder)).
You can also setup a local development environment with docker (see [section
Docker](#Docker)) or without docker (see [section Host](#Host)). To only view
the notebooks you can use the [nbviewer.org](https://nbviewer.org/) service
which can render the notebooks. Notebooks with outputs should be kept in a
separate branch to support viewing with outputs.


## Remote

Using a remote service is usually easy but gives less control. You need
uninterrupted access to the Internet. Exploring and running notebooks is
done through a web interface and your local browser.


### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fjenskmueller.com%2Fgit%2Ftemplate.git/master?urlpath=lab)

A [binder](https://mybinder.readthedocs.io/en/latest/) instance is
created remotely on demand when following this
[link](https://mybinder.org/v2/git/https%3A%2F%2Fjenskmueller.com%2Fgit%2Ftemplate.git/master?urlpath=lab)
(or when clicking the above badge) and destroyed after some inactivity
(see below). The instance is started in collaborative mode which
allows multiple users to work on the same notebooks using the same
JupyterLab instance. **No changes are saved persistently. You have to
download any changes you would like to keep.**

> Binder is meant for interactive and ephemeral interactive coding,
> meaning that it is ideally suited for relatively short sessions.
> Binder will automatically shut down user sessions that have more than
> 10 minutes of inactivity (if you leave a jupyterlab window open in the
> foreground, this will generally be counted as “activity”).

> Binder aims to provide up to six hours of session time per user
> session, or up to one cpu-hour for more computationally intensive
> sessions. Beyond that, we cannot guarantee that the session will
> remain running.

(from [How long will my Binder session
last?](https://mybinder.readthedocs.io/en/latest/about/about.html#how-long-will-my-binder-session-last))

A binder instance has at least 1GB and at most 2GB RAM and the kernel
will be restarted when using more than 2GB (see [How much memory am I
given when using
Binder?](https://mybinder.readthedocs.io/en/latest/about/about.html#user-memory)).

The instances are provided by Google Cloud, OVH, GESIS Notebooks and the
Turing Institute. Everything inside the Binder session is destroyed when
the session ends. Logs are kept for 30 days and anonymized IP adresses
are used for analytics.

> We take user privacy very seriously! Because Binder runs as a public,
> free service, we don’t require any kind of log-in that would let us
> keep track of user data. All code that is run, data analyzed, papers
> reproduced, classes taught - in short, everything that happens in a
> Binder session - is destroyed when the user logs off or becomes
> inactive for more than a few minutes.

> Here are the pieces of information we do keep: We run google analytics
> with anonymized IPs and no cookies, which gives us just enough
> information to know how Binder is being used, and but won’t be able to
> identify users. We also retain logs of IP addresses for 30 days, which
> is used solely in the case of detecting abuse of the service. If you
> have suggestions for how we can ensure the privacy of our data and
> users, we’d love to hear it!

(from [How does mybinder.org ensure user
privacy?](https://mybinder.readthedocs.io/en/latest/about/about.html?highlight=privacy#how-does-mybinder-org-ensure-user-privacy))


## Local

With a local setup everything is stored on your computer. The setup is
usally more involved and you require Internet access during the setup.
Later Internet access is only needed sporadically for certain tasks.

To get the repository you have to install [Git](https://git-scm.com/)
locally. Then you can clone the Git repository with

    $ git clone <repository URL>

This creates a folder with the name of the repository containing all
necessary files. The repository URL looks typically something like
`https://example.com/path/to/repository.git`. Change into the created
folder to explore the downloaded files.


### Docker

With Docker the development environment can be kept isolated from the
host system. Docker is supported on Windows, macOS, and Linux. Follow
[the offical instructions](https://docs.docker.com/get-docker/) to setup
docker first.

The image is defined by the [Dockerfile](Dockerfile) part of the cloned
repository. To build it locally, run

    $ docker-compose build --build-arg user_id=$(shell id -u)

This builds a Docker image containing all the necessary tools and
libraries for development.

    $ docker-compose up

This starts a containerized JupyterLab server which can be accessed from
the local machine (see the outputs of the above command).

The configuration for `docker-compose` is stored in
[docker-compose.yml](docker-compose.yml). Notably, it mounts the users
Git configuration `~/.gitconfig` and `.docker_bash_history` (for keeping
a history of executed commands) inside the Docker image. Both files have
to exist otherwise running the image with `docker-compose` will fail.
Also it mounts the local folder to allow making modifications from
within an running image.

Instead of JupyterLab you can also start a shell

    $ docker-compose run --service-ports --entrypoint "poetry shell" jupyterlab

inside the Docker container.

To clear the Docker build cache and remove its unused data, run

```
$ docker builder prune --all --force
$ docker system prune --volumes --force
```


### Host

You can also setup the work environment directly on your computer. For
this you need to have [Python](https://www.python.org/) and [GNU
Make](https://www.gnu.org/software/make/) installed.

First install [Poetry](https://python-poetry.org/)

    $ make install-poetry

which is used to manage Python dependencies. The dependencies are
installed by

    $ make install

according to [poetry.lock](poetry.lock) in a virtual environment under
`.venv/`. Note, these commands are also done in the
[Dockerfile](Dockerfile) to provide Debian Linux based container.

To start a shell in the environment, run

    $ make shell


# Development

Primarily, the development takes place in
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/). JupyterLab
provides a web-based interactive environment to run code and show its
outputs. To run various commands useful during development `Make`
targets are provided for convenience (see the [Makefile](Makefile) in
this repository).


## Starting JupyterLab

The remote approaches bring up a JupyterLab (or similar) web interface
directly. With a shell, you can to start it yourself by entering

    $ make jupyterlab

inside the virtual environment. The output shows a URL like
http://127.0.0.1:8888/lab?token=c8cc40c5d022bf648b3500545e2d403c38a4c0cb034dbdc6
which can be pasted into a browser's addressbar to access the just
started JupyterLab server. Within JupyterLab you can open, edit and run
code. Navigate to the `notebooks/` folder and open a notebook (files
ending in `.ipynb`). JupyterLab also provides a terminal which can be
used to enter commands. Go ahead to the [JupyterLab User
Guide](https://jupyterlab.readthedocs.io/en/stable/user/interface.html)
to familiarize yourself on how to use JupyterLab.


## Running all Notebooks

At times it can be useful to check that all notebooks in the
`notebooks/` folder are working properly. For convenience, enter

    $ make run-notebooks

for run all notebooks. Running all notebooks should only take a few
minutes.


## Notebooks Rendered in HTML

The target `html-notebooks` can be used to generate the notebooks
rendered as HTML files. Run,

    $ make html-notebooks

to generate a HTML file in the folder `html/` for each notebook in
`notebooks/`. The notebooks are executed. Hence, they include outputs.


## Code Style

In software projects one likes to make sure that the code is properly
formatted and organized. For this you can run

    $ make style-notebooks

which checks that all notebooks conform to the code style.


## Testing the Docker Image

The image for running on binder is tested via the target `test-image`.
Run

    $ make test-image

to build the image, run and style check the notebooks, and verify that
JupyterLab is started.


## Git

The [Git](https://git-scm.com/) version control system is used for
collaborative development and tracking software changes. Files can be
added and tracked by Git to keep a history of the project. It tells us
what changes were made when and ideally why.

To learn about the basics of Git, familiarize yourself with the
resources at https://git-scm.com/doc, specifically the quick reference
guides.

We provide a customized Git configuration to ease tracking and working
with notebooks which we recommend to set up via

    $ git config --local include.path ../.gitconfig

This adds the versioned [.gitconfig](.gitconfig) file to the Git
configuration. It sets up stripping of outputs for Jupyter notebooks in
the [notebooks/.gitattributes](notebooks/.gitattributes) file (since
notebook outputs can be annoying to track) and a Git pre-commit hook.
The pre-commit hook checks for trailing whitespace errors and performs
style checking of notebooks. It uses
[black](https://black.readthedocs.io/en/stable/),
[isort](https://pycqa.github.io/isort/) and
[flake8](https://flake8.pycqa.org/en/latest/) (see
[.githooks/pre-commit](.githooks/pre-commit)) to verify staged notebooks
when commiting. Also a template for commit messages is provided to help
with writing good commit messages (see [.gitmessage](.gitmessage)).


## Poetry

In case you need to add or update the Python dependencies use the
`poetry` command. For example new dependencies can be added by

    $ poetry add --dev <name-of-dependency>

This updates the poetry files [pyproject.toml](pyproject.toml) and
[poetry.lock](poetry.lock) to keep track of the added dependency. All
dependencies are installed in a virtual environment under `.venv/`.

More information on how to use poetry can be found at
https://python-poetry.org/docs/.


## Syncing Notebooks with Text Files

When you want to adapt notebooks in an editor or integrated development
environment (IDE) you can use the `sync` target which generates text
files in `sync/` corresponding to the notebooks in `notebooks/`. Newer
files in `sync/` (than the corresponding file in `notebooks`) are synced
back to the respective notebooks.

To sync notebooks and text files, run

    $ make sync

Beware of autosaving which may cause unintended syncs, i.e., assuming
you are editing a text file, saving it but then JupyterLab autosaves the
notebook. In this case your changes will be overwritten since the just
saved notebook is newer.
