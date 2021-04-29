
push!(LOAD_PATH, ".")
using Revise

using SIAC_filter, Printf

function l2_projection_legendre(p, z, w, f)
    J    = [basis.(Legendre, i) for i = 0:p]
    PLeg = hcat([J[i].(z) for i = 1:p+1]...)
    # mode = ( f, L_k ) / (L_k,L_k) ::: 1 / (L_k, L_k) = 0.5 * (2 k + 1)
    mode = [sum(w .* f .* PLeg[:,j]) * 0.5*(2.0*(j-1) + 1.0) for j = 1:p+1]
    proj = zeros(length(z))
    @einsum proj[k] = mode[j] * PLeg[k,j]
    return proj, vec(mode)
end

function filter_l2_projection(p,nE, showplots = false)

    Q  = 4 * p
    zQ, wQ = legendre(Q)
    omega = [-1, 1]
    h     = (omega[2] - omega[1]) / (nE)
    
    exac = zeros((nE , Q))
    proj = zeros((nE , Q))
    filt = zeros((nE , Q))
    t    = zeros((nE , Q))

    eL2   = zeros(2)
    eLInf = zeros(2)

    a = omega[1]

    l = p + 1
    r = p
    ker   = set_kernel(l,r)
    symcc = symmetric_pp(p,ker,zQ)
    kwide = Int(ceil(ker.support))
    # Total number of elements in the support (UNIFORM MESH)
    pwide = 2*kwide+1

    b     = omega[1]
    modes = zeros(nE,p+1)
    for e in 1:nE
    a         = b
    b         = a + h
    t[e,:]    = 0.5 .* ( (b - a) .* zQ .+ a .+ b)
    exac[e,:] = [ sinpi(t[e,i]) for i in 1:Q ]
    proj[e,:], modes[e,:] = l2_projection_legendre(p, zQ, wQ, exac[e,:])
    eL2[1]  += 0.5 * (b -a) * sum((exac[e,:] .- proj[e,:]).^2 .*wQ)
    end

    for e in 1:nE
    idx = zeros(Int64,pwide)
    for k =1:pwide
        kk2 = k-kwide-1
        idx[k] = e + kk2
        if idx[k] <= 0
            idx[k] += nE
        elseif idx[k] > nE
            idx[k] -= nE
        end
    end
    midx = modes[idx,:]
    locf = zeros(length(zQ))
    @einsum locf[j] = symcc[i,k,j] * midx[i,k]
    filt[e,:] = locf
    eL2[2]  += 0.5 * (b -a) * sum((exac[e,:] .- filt[e,:]).^2 .*wQ)
    end


    fP = vec(proj')
    fT = vec(t')
    fE = vec(exac')
    fF = vec(filt')

    eLInf[1] = maximum(abs.(fE .- fP))
    eLInf[2] = maximum(abs.(fE .- fF))

    eL2 = sqrt.(eL2)

    print("\n\nPROJETION ERROS ||u - uh||_0^2 = ", eL2[1], " ||u-uh||_inf ",eLInf[1],"\n")
    print("\n\nFILTERED  ERROS ||u - u*||_0^2 = ", eL2[2], " ||u-u*||_inf ",eLInf[2],"\n")
    if showplots == true
        pylab.subplot(1, 2, 1)
        pylab.title(" Curves ")
        pylab.plot(fT, fE, color = "green")
        pylab.plot(fT, fP, color = "red")
        pylab.plot(fT, fF, color = "orange")

        pylab.subplot(1, 2, 2)
        pylab.title(" Error ")
        pylab.plot(fT, fE - fP, color = "red")
        pylab.plot(fT, fE - fF, color = "orange")

        pylab.show()
    end
    return eL2, eLInf
end


nE0 = 10
nR  = 3

for p = 1:3
    showplots = true
    eL2   = zeros(nR + 1, 2)
    eLInf = zeros(nR + 1, 2)
    elmts = zeros(Int64,nR + 1)
    for k = 1:nR+1
        elmts[k] = 2 ^(k - 1) * nE0
        eL2[k,:],eLInf[k,:] = filter_l2_projection(p,elmts[k],showplots)
        showplots = false
    end
    print("\n\n RESULTS FOR p =",p,"\n")
    @printf("N\t ||u-uh||_0\t CR ||u-u*||_0\t CR ||u-uh||_inf\t CR ||u-u*||_inf\t CR\n")
    print("----------------------------------------------\n")
    for k in 1:nR+1
        if k == 1
            @printf("%5i\t %.5e\t NA\t %.5e\t NA\t %.5e\t NA\t %.5e\t NA\t\n"  , elmts[k], eL2[k,1],eL2[k,2],eLInf[k,1],eLInf[k,2])
        else
            r2 = log10.(eL2[k-1,:]./eL2[k,:])./log10(elmts[k] / elmts[k-1])
            ri = log10.(eLInf[k-1,:]./eLInf[k,:])./log10(elmts[k] / elmts[k-1])
            @printf("%5i\t %.5e\t %.2f\t %.5e\t %.2f\t %.5e\t %.2f\t %.5e\t %.2f\t\n",
                     elmts[k], eL2[k,1],r2[1], eL2[k,2],r2[2], eLInf[k,1],ri[1],eLInf[k,2],ri[2])
        end
        print("----------------------------------------------\n")
    end
end
