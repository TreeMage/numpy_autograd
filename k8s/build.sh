NAME="ls6-stud-registry.informatik.uni-wuerzburg.de/studkohlmann/prog-nn-autograd"
TAG="0.0.1"

cd .. && docker build -t"$NAME:$TAG" -f k8s/Dockerfile .