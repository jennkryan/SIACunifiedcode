push!(LOAD_PATH, ".")

module SIAC_filter

using Revise
using GaussQuadrature, Polynomials,SpecialPolynomials, PyCall, PyPlot
const SP = SpecialPolynomials

using BSplines: BSplineBasis,splinevalue

using Einsum, LinearAlgebra

pygui()

export test_p1_kernel, test_p2_kernel, symmetric_pp, set_kernel

mutable struct kernel
    l       :: Int
    r       :: Int
    breaks  :: Array{Float64,1}
    support :: Float64
    c_g     :: Array{Float64,1}
    splines :: BSplineBasis{}
end

function set_kernel(l, r)
    breaks = [-(r + 0.5 *l):(r + 0.5 *l);]
    spline = BSplineBasis(l,breaks)

    # Define matrix to determine kernel coefficients
    NS   = 2 * r + 1
    A    = zeros(NS,NS)
    b    = zeros(NS)
    b[1] = 1.0
    for m = 1:NS
        s1 = [sum([(-1.0)^(j+l-1) * binomial(l-1,j) *
                   ((j-0.5*(l -2.0) )^(l+n-1) - (j-0.5*l)^(l+n-1)) for j = 0:l-1]) for n = 1:m]
        for gam = 1:NS
            s2       = [binomial(m-1,n-1)*(gam-1-(NS - 1)*0.5)^(m-n)*factorial(n-1)/factorial(n-1+l) for n = 1:m]
            A[m,gam] = sum(s1 .* s2)
        end
    end
    c  = factorize(A)\b
    return kernel(l, NS, breaks,0.5 * (breaks[end] - breaks[1]), c, spline)
end

function map_std_coordinates(h,z)
    return vcat([0.5*( (h[j+1]-h[j]).*z.+(h[j+1]+h[j])) for j = 1:length(h)-1]...)
end

function plot_kernel(ker)
    z,_ = legendre(20)
    # Define B-spline breaks for a B-spline of order l
    x  = map_std_coordinates(ker.breaks, z)
    lx = length(x)
    bs = [splinevalue(ker.splines[m], x[j]) for m = ker.l:ker.l+ker.r-1, j = 1:lx]

    kerval = [sum(bs[:,j].*ker.c_g) for j = 1:lx]
    for j in 1:length(bs[:,1])
        plot(x,bs[j,:], c ="r", ls = "-")
    end
    plot(x,kerval,  c ="black", ls = "--")

    return nothing
end

function integral_ab(ahat,bhat,p,kwide,pwide,ker,kk,centre)
    # Evaluate the pp integrals using Gauss-Legendre quadrature.
    # The integral is: int_a^b K(0.5(centre - x) - kk)P^(m)(x) dx,
    # where K kernel, centre evaluation point, and P^(m) m-deg Legendre pol

    # Approximation is a polynomial of degree p, kernel is a
    # polynomial of degree l-1  ==> we need p+l-1 = 2gpts-1, where n is the number of points.
    # Hence, gpts=(p+l)/2.  If p+l is odd, we want the first integer >= (p+l)/2, hence the
    # ceiling function.
    z,w  = legendre(Int(ceil(0.5*(p+ker.l))))


    # Ensure integration does not go beyond the support of the kernel.
    intlow = centre-2.0*(ker.support+kk)
    intup  = centre+2.0*(ker.support-kk)
    ahat   = max(ahat, intlow)
    bhat   = min(bhat, intup)
    # only perform the integration if ahat < bhat
    if ahat > bhat
        return zeros(p+1)
    end

    # scale the integration interval to (-1,1) to use Gauss-Legendre quadrature
    zeta = map_std_coordinates([ahat,bhat],z)

    # Evaluation coordinate for the kernel integration
    kerzeta = 0.5.*(centre.-zeta).-Float64(kk)
    bs   = [splinevalue(ker.splines[m], kerzeta[j])
            for m = ker.l:ker.l+ker.r-1, j = 1:length(kerzeta)]
    fker = [sum(bs[:,j].*ker.c_g) for j = 1:length(kerzeta)]
    J    = [basis.(Legendre, i) for i = 0:p]
    PLeg = [J[i].(zeta) for i = 1:p+1]
    intj = [sum(fker.* PLeg[m].*w) for m = 1:p+1]

    # Obtain the integral value
    return vec(0.5 *(bhat-ahat) .* [sum(fker.* PLeg[m].*w) for m = 1:p+1])
end

function exactp1convcoeff(zEval)
    exsym0 = zeros(5,2,length(zEval))

    exsym0[1,1,:] = .-(-1.0 .+ zEval).^ 2 ./ 96.0
    exsym0[1,2,:] = -0.1e1 / 0.144e3 .- zEval.^ 3 / 288 .+ zEval ./ 96

    exsym0[2,1,:] = 0.1e1 / 0.12e2 .- 0.7e1 / 0.24e2 .* zEval .+ zEval.^2 ./ 6
    exsym0[2,2,:] = 0.7e1 / 0.72e2 .- zEval ./ 6 + zEval.^ 3 ./ 18

    exsym0[3,1,:] = 0.41e2 / 0.48e2 .- 0.5e1 / 0.16e2 .* zEval.^2
    exsym0[3,2,:] = .5e1 / 0.16e2 .* zEval .- 0.5e1 / 0.48e2 .* zEval.^ 3

    exsym0[4,1,:] =  0.7e1 / 0.24e2 .* zEval .+ 0.1e1 / 0.12e2 .+ zEval.^2 ./ 6
    exsym0[4,2,:] = -0.7e1 / 0.72e2 .- zEval ./ 6 .+ zEval.^3 ./ 18

    exsym0[5,1,:] = .-(zEval .+ 1).^ 2 ./ 96
    exsym0[5,2,:] = zEval ./ 96 .+ 0.1e1 / 0.144e3 .- zEval.^3 ./ 288

    return exsym0
end

function exactp2convcoeff(zEval)
    n      = length(zEval)
    exsym0 = zeros(9,3,n)
    for j = 1:n
        Zeta = zEval[j]
        if Zeta < 0
            exsym0[1,1,j] = -37/92160 * Zeta ^3
            exsym0[1,2,j] = -37/368640 * Zeta ^3 * (Zeta + 4)
            exsym0[1,3,j] = -37/614400 * Zeta ^5 - 37/122880*Zeta ^4 - 37/92160 * Zeta ^3

            exsym0[2,1,j] = 499/92160 * Zeta ^3 + 37/11520 + 37/15360 * Zeta ^2 - 37/7680 * Zeta
            exsym0[2,2,j] = 37/23040 - 37/23040 * Zeta + 499/368640 * Zeta ^4 + 499/92160 * Zeta ^3
            exsym0[2,3,j] = 499/614400 * Zeta ^5 + 499/122880 * Zeta ^4 + 499/92160 * Zeta ^3 + 37/115200

            exsym0[3,1,j] = -77/2560 * Zeta ^2 + 97/1920 * Zeta - 1/48 - 433/10240 * Zeta ^3
            exsym0[3,2,j] = -97/5760 + 77/3840 * Zeta - 433/40960 * Zeta ^4 -433/10240 * Zeta ^3
            exsym0[3,3,j] = -1299/204800 * Zeta ^5 - 1299/40960 * Zeta ^4 - 433/10240 * Zeta ^3 - 77/19200

            exsym0[4,1,j] = 229/1024 * Zeta ^2 - 517/1536 * Zeta + 123/1280 + 1891/18432 * Zeta ^3
            exsym0[4,2,j] = 517 / 4608 - 229/ 1536 * Zeta + 1891/73728 * Zeta ^4 + 1891/18432 * Zeta ^3
            exsym0[4,3,j] = 1891/122880 * Zeta ^5 + 1891/24576 * Zeta ^4 + 1891/18432 * Zeta ^3 + 229/7680

            exsym0[5,1,j] = -1891/18432 * Zeta ^3 - 301/768 * Zeta ^2 + 607/720
            exsym0[5,2,j] = -1891/73728 * Zeta * (Zeta ^3 + 4 * Zeta ^2 - 19264/1891)
            exsym0[5,3,j] = -301/5760 - 1891/122880 * Zeta ^5 - 1891/18432 * Zeta ^3 - 1891/24576 * Zeta ^4

            exsym0[6,1,j] = 229/1024 * Zeta ^2 + 517/1536 * Zeta + 123/1280 + 433/10240 * Zeta ^3
            exsym0[6,2,j] = -517/4608 - 229/1536 * Zeta + 433/40960 * Zeta ^4 + 433/10240 * Zeta ^3
            exsym0[6,3,j] = 229/7680 + 1299/204800 * Zeta ^5 + 1299/40960 * Zeta ^4 + 433/10240 * Zeta ^3

            exsym0[7,1,j] = -0.77e2 / 0.2560e4 * Zeta ^2 - 0.97e2 / 0.1920e4 * Zeta - 0.1e1 / 0.48e2 - 0.499e3 / 0.92160e5 * Zeta ^3
            exsym0[7,2,j] = 0.97e2 / 0.5760e4 + 0.77e2 / 0.3840e4 * Zeta - 0.499e3 / 0.368640e6 * Zeta ^4 - 0.499e3 / 0.92160e5 * Zeta ^3
            exsym0[7,3,j] = -0.499e3 / 0.614400e6 * Zeta ^5 - 0.499e3 / 0.122880e6 * Zeta ^4 - 0.499e3 / 0.92160e5 * Zeta ^3 - 0.77e2 / 0.19200e5

            exsym0[8,1,j] = 0.37e2 / 0.92160e5 * (Zeta + 2) ^3
            exsym0[8,2,j] = 0.37e2 / 0.368640e6 * (Zeta - 2) * (Zeta + 2) ^3
            exsym0[8,3,j] = 0.37e2 / 0.115200e6 + 0.37e2 / 0.614400e6 * Zeta ^5 + 0.37e2 / 0.122880e6 * Zeta ^4 + 0.37e2 / 0.92160e5 * Zeta ^3

        elseif Zeta>0
                exsym0[2,1,j] = -0.37e2 / 0.92160e5 * (Zeta - 2) ^3
                exsym0[2,2,j] = -0.37e2 / 0.368640e6 * (Zeta + 2) * (Zeta - 2) ^3
                exsym0[2,3,j] = -0.37e2 / 0.614400e6 * Zeta ^5 + 0.37e2 / 0.122880e6 * Zeta ^4 - 0.37e2 / 0.92160e5 * Zeta ^3 + 0.37e2 / 0.115200e6

                exsym0[3,1,j] = 0.499e3 / 0.92160e5 * Zeta ^3 - 0.77e2 / 0.2560e4 * Zeta ^2 + 0.97e2 / 0.1920e4 * Zeta - 0.1e1 / 0.48e2
                exsym0[3,2,j] = 0.499e3 / 0.368640e6 * Zeta ^4 - 0.499e3 / 0.92160e5 * Zeta ^3 + 0.77e2 / 0.3840e4 * Zeta - 0.97e2 / 0.5760e4
                exsym0[3,3,j] = 0.499e3 / 0.614400e6 * Zeta ^5 - 0.499e3 / 0.122880e6 * Zeta ^4 + 0.499e3 / 0.92160e5 * Zeta ^3 - 0.77e2 / 0.19200e5

                exsym0[4,1,j] = -0.433e3 / 0.10240e5 * Zeta ^3 + 0.229e3 / 0.1024e4 * Zeta ^2 - 0.517e3 / 0.1536e4 * Zeta + 0.123e3 / 0.1280e4
                exsym0[4,2,j] = -0.433e3 / 0.40960e5 * Zeta ^4 + 0.433e3 / 0.10240e5 * Zeta ^3 - 0.229e3 / 0.1536e4 * Zeta + 0.517e3 / 0.4608e4
                exsym0[4,3,j] = 0.229e3 / 0.7680e4 - 0.1299e4 / 0.204800e6 * Zeta ^5 + 0.1299e4 / 0.40960e5 * Zeta ^4 - 0.433e3 / 0.10240e5 * Zeta ^3


                exsym0[5,1,j] = 0.1891e4 / 0.18432e5 * Zeta ^3 - 0.301e3 / 0.768e3 * Zeta ^2 + 0.607e3 / 0.720e3
                exsym0[5,2,j] = 0.1891e4 / 0.73728e5 * Zeta ^4 - 0.1891e4 / 0.18432e5 * Zeta ^3 + 0.301e3 / 0.1152e4 * Zeta
                exsym0[5,3,j] = -0.301e3 / 0.5760e4 + 0.1891e4 / 0.18432e5 * Zeta ^3 + 0.1891e4 / 0.122880e6 * Zeta ^5 - 0.1891e4 / 0.24576e5 * Zeta ^4

                exsym0[6,1,j] = -0.1891e4 / 0.18432e5 * Zeta ^3 + 0.229e3 / 0.1024e4 * Zeta ^2 + 0.517e3 / 0.1536e4 * Zeta + 0.123e3 / 0.1280e4
                exsym0[6,2,j] = -0.1891e4 / 0.73728e5 * Zeta ^4 + 0.1891e4 / 0.18432e5 * Zeta ^3 - 0.229e3 / 0.1536e4 * Zeta - 0.517e3 / 0.4608e4
                exsym0[6,3,j] = 0.229e3 / 0.7680e4 - 0.1891e4 / 0.18432e5 * Zeta ^3 - 0.1891e4 / 0.122880e6 * Zeta ^5 + 0.1891e4 / 0.24576e5 * Zeta ^4

                exsym0[7,1,j] = 0.433e3 / 0.10240e5 * Zeta ^3 - 0.77e2 / 0.2560e4 * Zeta ^2 - 0.97e2 / 0.1920e4 * Zeta - 0.1e1 / 0.48e2
                exsym0[7,2,j] = 0.433e3 / 0.40960e5 * Zeta ^4 - 0.433e3 / 0.10240e5 * Zeta ^3 + 0.77e2 / 0.3840e4 * Zeta + 0.97e2 / 0.5760e4
                exsym0[7,3,j] = 0.1299e4 / 0.204800e6 * Zeta ^5 - 0.1299e4 / 0.40960e5 * Zeta ^4 + 0.433e3 / 0.10240e5 * Zeta ^3 - 0.77e2 / 0.19200e5

                exsym0[8,1,j] = -0.499e3 / 0.92160e5 * Zeta ^3 + 0.37e2 / 0.15360e5 * Zeta ^2 + 0.37e2 / 0.7680e4 * Zeta + 0.37e2 / 0.11520e5
                exsym0[8,2,j] = -0.499e3 / 0.368640e6 * Zeta ^4 + 0.499e3 / 0.92160e5 * Zeta ^3 - 0.37e2 / 0.23040e5 * Zeta - 0.37e2 / 0.23040e5
                exsym0[8,3,j] = -0.499e3 / 0.92160e5 * Zeta ^3 - 0.499e3 / 0.614400e6 * Zeta ^5 + 0.499e3 / 0.122880e6 * Zeta ^4 + 0.37e2 / 0.115200e6

                exsym0[9,1,j] = 0.37e2 / 0.92160e5 * Zeta ^3
                exsym0[9,2,j] = 0.37e2 / 0.368640e6 * Zeta ^3 * (Zeta - 4)
                exsym0[9,3,j] = 0.37e2 / 0.614400e6 * Zeta ^5 - 0.37e2 / 0.122880e6 * Zeta ^4 + 0.37e2 / 0.92160e5 * Zeta ^3
        else
                exsym0[2,1,j] = 0.37e2 / 0.11520e5
                exsym0[2,2,j] = 0.37e2 / 0.23040e5
                exsym0[2,3,j] = 0.37e2 / 0.115200e6

                exsym0[3,1,j] = -0.1e1 / 0.48e2
                exsym0[3,2,j] = -0.97e2 / 0.5760e4
                exsym0[3,3,j] = -0.77e2 / 0.19200e5

                exsym0[4,1,j] = 0.123e3 / 0.1280e4
                exsym0[4,2,j] = 0.517e3 / 0.4608e4
                exsym0[4,3,j] = 0.229e3 / 0.7680e4

                exsym0[5,1,j] =  0.607e3 / 0.720e3
                exsym0[5,2,j] = 0.0
                exsym0[5,3,j] = -0.301e3 / 0.5760e4

                exsym0[6,1,j] = 0.123e3 / 0.1280e4
                exsym0[6,2,j] = -0.517e3 / 0.4608e4
                exsym0[6,3,j] = 0.229e3 / 0.7680e4

                exsym0[7,1,j] = -0.1e1 / 0.48e2
                exsym0[7,2,j] = 0.97e2 / 0.5760e4
                exsym0[7,3,j] = -0.77e2 / 0.19200e5

                exsym0[8,1,j] = 0.37e2 / 0.11520e5
                exsym0[8,2,j] = -0.37e2 / 0.23040e5
                exsym0[8,3,j] = 0.37e2 / 0.115200e6
        end
    end
    return exsym0
end

function symmetric_pp(p, ker, x)
    nx = length(x)
    # Post-processor support is (xbar - kernelsupp*dx, xbar + kernelsupp*dx)

    # Make the element counter and integer value
    kwide = Int(ceil(ker.support))

    # Total number of elements in the support (UNIFORM MESH)
    pwide = 2*kwide+1

    # Accoiunt case B-spline breaks are not aligned with the evaluation point.
    # This occurs with l is odd (odd B-spline order)
    kres = Float64(ker.l%2)
    # symcc is the symmetric post-processing matrix
    symcc = zeros(pwide,p+1,nx)
    #appr = p, kernel = l-1, need p+l-1 = 2*Q-1, Q= ceil(p+l)/2.(p+l can be odd)
    #Evaluate the pp integrals using Gauss-Legendre quadrature.
    z,w  = legendre(Int(ceil(0.5*(p+ker.l))))
    for j =1:nx
        if kres != 0 && x[j] > 0.0  kres = -1.0
        end
        zetaEval = x[j] + kres
        for kk1 = 1:pwide
            kk = kk1-kwide -1
            # Integral evaluation arrays
            if abs(x[j]) < 1.e-33
                ahat = fill(-1.0,1)
                bhat = fill(1.0,1)
            else
                ahat = [-1.0 zetaEval]
                bhat = [zetaEval 1.0]
            end
            intj = hcat([integral_ab(ahat[k],bhat[k],p,kwide,pwide,ker,kk,x[j]) for k = 1:length(ahat)]...)
            symcc[kk1,:,j] = [0.5 .* sum(intj[m,:]) for m = 1:p+1]
            continue
            # The integral is: int_a^b K(0.5(centre - x) - kk)P^(m)(x) dx,
            # K kernel, centre: eval point, and P^(m) m-deg Legendre pol
            intlow = x[j]-2.0*(ker.support+kk)
            intup  = x[j]+2.0*(ker.support-kk)
            for k = 1:length(ahat)
                # Ensure integration doesn't go beyond the support of the kernel
                ak     = max(ahat[k], intlow)
                bk     = min(bhat[k], intup)
                # only perform the integration if ahat < bhat
                if ak >= bk
                    continue
                end
                # scale to (-1,1) to use Gauss-Legendre quadrature
                zeta = map_std_coordinates([ak,bk], z)
                # Evaluation coordinate for the kernel integration
                kerzeta = 0.5.*(x[j].-zeta).-Float64(kk)
                # Obtain the splines value at the gauss points
                bs   = [splinevalue(ker.splines[m], kerzeta[i])
                        for m = ker.l:ker.l+ker.r-1, i = 1:length(kerzeta)]
                # Evaluate kernel sum c_g * splines
                fker = [sum(bs[:,i].*ker.c_g) for i = 1:length(kerzeta)]
                J    = [basis.(Legendre, i) for i = 0:p]
                PLeg = [J[i].(zeta) for i = 1:p+1]
                # Obtain the integral value
                dx = 0.5 * (0.5 * (bk-ak))
                symcc[kk1,:,j] += [dx.* sum(fker.* PLeg[m].*w) for m = 1:p+1]
            end
        end
    end
    return symcc
end

function dg_breaks_equal_bsbreaks(p,kwide,pwide,L,z,w,ker,zEvalj)
    gp      = length(z)
    kerzeta = zeros(gp)

    J       = [basis.(Legendre, i) for i = 0:p]
    y       = 0.5*L.*z
    pLeg    = [J[i].(y) for i = 1:p+1]

    for kk1 = 1:pwide
        if kk1 != 0 && kk1 !=0 pwide-1
            kk      = Float64(kk1-kwide)
            ahat    = -1.0
            bhat    = 1.0
            kerzeta = (zEvalj.-z)./L.-kk
            fker    = evalkernel(ker,gp,kerzeta)
            xintsum[kk1,:] = [sum(fker.*pLeg[m,:].*w) for m = 1:p+1]
        end
    end
    return xintsum
end

function test_p1_kernel()
    zw = legendre(2)
    l = 2
    p = 1
    m = 1
    ker   = set_kernel(l,m)
    conv1 = symmetric_pp(p,ker,zw[1])
    conv2 = exactp1convcoeff(zw[1])
    print("\n\n CONVOLUTION p1 ---> ERROR BETWEEN EXACT AND COMPUTED ", sum(abs.(conv1 .- conv2)),"\n")
    plot_kernel(ker)
end

function test_p2_kernel()
    zw = legendre(2)
    l = 3
    p = 2
    m = 2
    ker = set_kernel(l,m)
    conv1 = symmetric_pp(p,ker,zw[1])
    conv2 = exactp2convcoeff(zw[1])
    print("\n CONVOLUTION p2 ---> ERROR BETWEEN EXACT AND COMPUTED ", sum(abs.(conv1 .- conv2)),"\n")
    plot_kernel(ker)
end

end
