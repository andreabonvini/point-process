# Gradient vector & Hessian matrix

For now let's consider the case without right-censoring.
$$
\mu(H_{u_{i}},\theta_t)=\theta_0+\sum_{j=1}^{n-1}\theta_ju_{i-j+1}
$$

$$
p(u_i|k,\theta_t) = \left[\frac{k}{2\pi\cdot{u_i}^3}\right]^{1/2}e^{-\frac{1}{2}\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}}
$$

$$
\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) = -\sum_{i=0}^{m-1}\eta_i\log p({u}_i|k,\theta_t)\\
$$

## Gradient `Vector`

$$
\nabla\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)=
\begin{bmatrix}
\frac{\part}{\part k}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)\\
\frac{\part}{\part \eta_0}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)\\
\vdots\\
\frac{\part}{\part \eta_{m-1}}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)\\
\frac{\part}{\part \theta_0}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)\\
\vdots\\
\frac{\part}{\part \theta_{n-1}}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)
\end{bmatrix}
$$

For $k$ we have: 
$$
\frac{\part}{\part k}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part k}\left(-\sum_{i=0}^{m-1}\eta_i\log p(u_i|k,\theta_t)\right) = \\

\frac{\part}{\part k}
\left(
-
\sum_{i=0}^{m-1}
\eta_i
\log
\left(
\left[\frac{k}{2\pi\cdot{u_i}^3}\right]^{1/2}e^{-\frac{1}{2}\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}}
\right)
\right)=\\

\frac{\part}{\part k}
\left(
\sum_{i=0}^{m-1}
-
\frac{1}{2}
\eta_i
\log
\left(
\left[\frac{k}{2\pi\cdot{u_i}^3}\right]
\right)
+\frac{1}{2}
\eta_i
\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}
\right)=\\

\frac{\part}{\part k}
\left(
\sum_{i=0}^{m-1}
-
\frac{1}{2}
\eta_i
\log
\left(
k
\right)
+\eta_i\log
\left(2\pi\cdot{u_i}^3\right)
+\frac{1}{2}
\eta_i
\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}
\right)=\\

\color{blue}
{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}
\left(-\frac{1}{k}+\frac{\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}\right)
}
$$
For $\eta_j$ we have:
$$
\frac{\part}{\part \eta_j}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\
\frac{\part}{\part \eta_j}\left(-\sum_{i=0}^{m-1}\eta_i\log p({u}_i|k,\theta_t)\right)=\\

\color{blue}
{
-\log p(u_j|k,\theta_t)
}
$$
For $\theta_j$ we have:
$$
\frac{\part}{\part \theta_j}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part \theta_j}\left(-\sum_{i=0}^{m-1}\eta_i\log p(u_i|k,\theta_t)\right) = \\

\frac{\part}{\part \theta_j}
\left(
-
\sum_{i=0}^{m-1}
\eta_i
\log
\left(
\left[\frac{k}{2\pi\cdot{u_i}^3}\right]^{1/2}e^{-\frac{1}{2}\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}}
\right)
\right)=\\

\frac{\part}{\part \theta_j}
\left(
\sum_{i=0}^{m-1}
-
\frac{1}{2}
\eta_i
\log
\left(
\left[\frac{k}{2\pi\cdot{u_i}^3}\right]
\right)
+\frac{1}{2}
\eta_i
\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}
\right)=\\

\frac{\part}{\part \theta_j}
\left(
\sum_{i=0}^{m-1}
-
\frac{1}{2}
\eta_i
\log
\left(
k
\right)
+\eta_i\log
\left(2\pi\cdot{u_i}^3\right)
+\frac{1}{2}
\eta_i
\frac{k\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}
\right)=\\


\frac{\part}{\part \theta_j}
\left(
\sum_{i=0}^{m-1}
\frac{\eta_ik}{2u_i}
\frac
{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2
}
{
\mu(H_{u_{i-1}},\theta_t)^2
}
\right)=\\


\sum_{i=0}^{m-1}
\frac{\eta_ik}{2u_i}
\cdot
\frac{
2\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(-u_{i-j})\cdot\mu(H_{u_{i-1}},\theta_t)^2
-
2\mu(H_{u_{i-1}},\theta_t)(u_{i-j})\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2
}
{\mu(H_{u_{i-1}},\theta_t)^4}=\\


\sum_{i=0}^{m-1}
\frac{\eta_ik}{u_i}
\cdot
\frac{
-\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})\cdot\mu(H_{u_{i-1}},\theta_t)
-
(u_{i-j})\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2
}
{\mu(H_{u_{i-1}},\theta_t)^3}=\\


\sum_{i=0}^{m-1}
\frac{\eta_ik}{u_i}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
\left(
-\mu(H_{u_{i-1}},\theta_t)
-u_i
+\mu(H_{u_{i-1}},\theta_t)
\right)
}
{\mu(H_{u_{i-1}},\theta_t)^3}=\\


\sum_{i=0}^{m-1}
\frac{\eta_ik}{u_i}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
\left(
-\mu(H_{u_{i-1}},\theta_t)
-u_i
+\mu(H_{u_{i-1}},\theta_t)
\right)
}
{\mu(H_{u_{i-1}},\theta_t)^3}=\\


-\sum_{i=0}^{m-1}
\frac{\eta_ik}{u_i}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
\left(
u_i
\right)
}
{\mu(H_{u_{i-1}},\theta_t)^3}=\\

\color{blue}
{
-\sum_{i=0}^{m-1}
{\eta_ik}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
}
$$
Since $\mu(H_{u_{i-1}},\theta_t)=\theta_0+\sum_{j=1}^{n-1}\theta_ju_{i-j}$  we note that in case $\color{black}{j=0}$, we must have that $u_{i-j} = u_i =1$

(`Note: In code this case is handled by the fact the u[i] = 1`)

## Hessian `Matrix`

$$
\nabla^2\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t)=

\begin{bmatrix}
\frac{\part^2}{\part^2 k}\mathcal{L}(\cdot)
&
\frac{\part}{\part k\part \eta_0}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part k\part\eta_{m-1}}\mathcal{L}(\cdot)
&
\frac{\part}{\part k\part\theta_0}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part k\part\theta_{n-1}}\mathcal{L}(\cdot)\\


\cdot
&
\frac{\part^2}{\part^2 \eta_0}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part \eta_0\part\eta_{m-1}}\mathcal{L}(\cdot)
&
\frac{\part}{\part \eta_0\part\theta_{0}}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part \eta_0\part\theta_{n-1}}\mathcal{L}(\cdot)
\\

\vdots
&
\vdots
&
\ddots
&
\vdots
&
\vdots
&
\vdots
&
\vdots
\\

\cdot &
\cdot &
\cdots &
\frac{\part^2}{\part^2 \eta_{m-1}}\mathcal{L}(\cdot)
&
\frac{\part}{\part \eta_{m-1}\part\theta_0}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part \eta_{m-1}\part\theta_{n-1}}\mathcal{L}(\cdot)
\\

\cdot &\cdot &\cdots & \cdot &
\frac{\part^2}{\part^2 \theta_0}\mathcal{L}(\cdot)
&
\cdots
&
\frac{\part}{\part \theta_0\theta_{n-1}}\mathcal{L}(\cdot)
\\


\cdot & \cdot & \vdots & \cdot & \cdot & \ddots & \vdots

\\

\cdot & \cdot & \cdots & \cdot & \cdot & \cdots &
\frac{\part^2}{\part^2 \theta_{n-1}}\mathcal{L}(\cdot)
\end{bmatrix}
$$

### $k$

For $k$ we have:
$$
\frac{\part^2}{\part^2 k}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part k}
\left(
{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}
\left(-\frac{1}{k}+\frac{\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}\right)
}
\right)=\\


\frac{\part}{\part k}
\left(
-
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}k^{-1}
\right)=\\


\color{blue}{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}k^{-2}
}
$$

$$
\frac{\part}{\part k\part\eta_{j}}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part \eta_j}
\left(
{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}
\left(-\frac{1}{k}+\frac{\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}\right)
}
\right)=\\

\frac{\part}{\part \eta_j}
\left(
{
\frac{\eta_j}{2}
\left(-\frac{1}{k}+\frac{\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}\right)
}
\right)=\\



\color{blue}
{
\frac{1}{2}
\left(-\frac{1}{k}+\frac{\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}\right)
}
$$

$$
\frac{\part}{\part k\part\theta_{j}}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\


\frac{\part}{\part \theta_j}
\left(
{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2}
\left(-\frac{1}{k}+\frac{\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2\cdot u_i}\right)
}
\right)=\\


\frac{\part}{\part \theta_j}
\left(
{
\sum_{i=0}^{m-1}
\frac{\eta_i}{2u_i}
\frac{\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)^2}{\mu(H_{u_{i-1}},\theta_t)^2}
}
\right)=\\
\cdots\\
\color{blue}
{
-\sum_{i=0}^{m-1}
{\eta_i}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
}
$$

### $\eta$

For $\eta_j$ we have:
$$
\frac{\part}{\part \eta_j\part k}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part k}\left(-\log p(u_j|k,\theta_t)\right) = \\


\frac{\part}{\part k}
\left(
-
\log
\left(
\left[\frac{k}{2\pi\cdot{u_j}^3}\right]^{1/2}e^{-\frac{1}{2}\frac{k\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}}
\right)
\right)=\\

\frac{\part}{\part k}
\left(
-
\frac{1}{2}
\log
\left(
\frac{k}{2\pi\cdot{u_j}^3}
\right)
+\frac{1}{2}
\frac{k\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}
\right)=\\


\frac{\part}{\part k}
\left(
-
\frac{1}{2}
\log k
+ \frac{1}{2}\log \left({2\pi\cdot{u_j}^3}\right)
+\frac{1}{2}
\frac{k\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}
\right)=\\

\color{purple}
{
\frac{1}{2}
\left(-\frac{1}{k}+\frac{\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}\right)
}
$$
We have that $\frac{\part}{\part \eta_j\part k}\mathcal{L}(\cdot) = \frac{\part}{\part k\part \eta_j}\mathcal{L}(\cdot)$ as *expected*.
$$
\frac{\part^2}{\part^2 \eta_j}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\
\frac{\part}{\part \eta_j}\left(-\log p(u_j|k,\theta_t)\right) = \color{blue}{0}
$$

$$
\frac{\part}{\part \eta_j\eta_q}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\
\frac{\part}{\part \eta_q}\left(-\log p(u_j|k,\theta_t)\right) = \color{blue}{0}
$$

$$
\frac{\part}{\part \eta_j\theta_q}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\


\frac{\part}{\part \theta_q}\left(-\log p(u_j|k,\theta_t)\right) = \\


\frac{\part}{\part \theta_q}
\left(
-
\frac{1}{2}
\log k
+ \frac{1}{2}\log \left({2\pi\cdot{u_j}^3}\right)
+\frac{1}{2}
\frac{k\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2\cdot u_j}
\right)=\\


\frac{\part}{\part \theta_q}
\left(

\frac{k}{2u_j}
\frac{\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)^2}{\mu(H_{u_{j-1}},\theta_t)^2}
\right)=\\

\color{blue}
{
-k
\cdot
\frac{
\left(u_j-\mu(H_{u_{j-1}},\theta_t)\right)(u_{j-q})
}
{\mu(H_{u_{j-1}},\theta_t)^3} 
}
$$

### $\theta$

For $\theta_j$ we have:
$$
\frac{\part}{\part \theta_j\part k}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part k}
\left(
-\sum_{i=0}^{m-1}
{\eta_ik}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)
=\\


\color{purple}
{
-\sum_{i=0}^{m-1}
{\eta_i}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
}
$$
We have that $\frac{\part}{\part \theta_j\part k}\mathcal{L}(\cdot) = \frac{\part}{\part k\part \theta_j}\mathcal{L}(\cdot)$ as *expected*.
$$
\frac{\part}{\part \theta_j\part \eta_q}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part \eta_q}
\left(
-\sum_{i=0}^{m-1}
{\eta_ik}
\cdot
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)
=\\

\color{purple}
{
-
k
\cdot
\frac{
\left(u_q-\mu(H_{u_{q-1}},\theta_t)\right)(u_{q-j})
}
{\mu(H_{u_{q-1}},\theta_t)^3}
}
$$
We have that $\frac{\part}{\part \theta_j\part \eta_q}\mathcal{L}(\cdot) = \frac{\part}{\part \eta_q\part \theta_j}\mathcal{L}(\cdot)$ as *expected*.
$$
\frac{\part^2}{\part^2 \theta_j}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\

\frac{\part}{\part \theta_j}
\left(
-\sum_{i=0}^{m-1}\eta_i{k}\cdot
\left(
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)\cdot(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)
\right)=\\

-\sum_{i=0}^{m-1}\eta_i{k}
(u_{i-j})\cdot
\frac{\part}{\part \theta_j}
\left(
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)\
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)=\\

-\sum_{i=0}^{m-1}\eta_i{k}
(u_{i-j})
\cdot
\left(
\frac{
(-u_{i-j})\cdot\mu(H_{u_{i-1}},\theta_t)^3-3\mu(H_{u_{i-1}},\theta_t)^2\cdot u_{i-j}\cdot\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)
}
{\mu(H_{u_{i-1}},\theta_t)^6}
\right)0\\


-\sum_{i=0}^{m-1}\eta_i{k}
(u_{i-j})
\cdot
\left(
\frac{
(-u_{i-j})\cdot\mu(H_{u_{i-1}},\theta_t)-3\cdot u_{i-j}\cdot\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)\
}
{\mu(H_{u_{i-1}},\theta_t)^4}
\right)=\\



-\sum_{i=0}^{m-1}\eta_i{k}
(u_{i-j})
\cdot
\left(
\frac{
-
3\cdot u_{i-j}\cdot u_i
+
2\cdot u_{i-j}
\cdot
\mu(H_{u_{i-1}},\theta_t)
}
{\mu(H_{u_{i-1}},\theta_t)^4}
\right)=\\


\color{blue}
{
\sum_{i=0}^{m-1}\eta_i{k}(u_{i-j})^2\cdot
\left(
\frac{
3\cdot u_i
-
2 \cdot \mu(H_{u_{i-1}},\theta_t)
}
{\mu(H_{u_{i-1}},\theta_t)^4}
\right)
}
$$

$$
\frac{\part}{\part \theta_j \part \theta_q}\mathcal{L}(u_{t-m:t}|k,\mathbf{\eta}_t,\theta_t) =\\


\frac{\part}{\part \theta_q}
\left(
-\sum_{i=0}^{m-1}\eta_i{k}\cdot
\left(
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)\cdot(u_{i-j})
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)
\right)=\\


-\sum_{i=0}^{m-1}\eta_i{k}
(u_{i-j})\cdot
\frac{\part}{\part \theta_q}
\left(
\frac{
\left(u_i-\mu(H_{u_{i-1}},\theta_t)\right)\
}
{\mu(H_{u_{i-1}},\theta_t)^3}
\right)=\\


\color{blue}
{
\sum_{i=0}^{m-1}\eta_i{k}(u_{i-j})(u_{i-q})\cdot
\left(
\frac{
3\cdot u_i
-
2 \cdot \mu(H_{u_{i-1}},\theta_t)
}
{\mu(H_{u_{i-1}},\theta_t)^4}
\right)
}
$$

