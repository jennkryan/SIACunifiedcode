push!(LOAD_PATH, ".")
using Revise
using SIAC_filter, Einsum, Printf, SpecialPolynomials, Polynomials, GaussQuadrature, PyCall, PyPlot
const SP = SpecialPolynomials
pygui()

function l2_projection_legendre(p, z, w, f)
    J    = [basis.(Legendre, i) for i = 0:p]
    PLeg = hcat([J[i].(z) for i = 1:p+1]...)
    # mode = ( f, L_k ) / (L_k,L_k) ::: 1 / (L_k, L_k) = 0.5 * (2 k + 1)
    mode = [sum(w .* f .* PLeg[:,j]) * 0.5*(2.0*(j-1) + 1.0) for j = 1:p+1]
    proj = zeros(length(z))
    @einsum proj[k] = mode[j] * PLeg[k,j]
    return proj, vec(mode)
end

function filter_l2_projection(p,nE, ax = nothing)

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

    ymin = maximum(abs.(fE .- fP))
    eL2 = sqrt.(eL2)

    if ax != nothing
        ax[1].plot(fT, abs.(fE .- fP), label = "N="*string(nE))
        ax[2].plot(fT, abs.(fE .- fF), label = "N="*string(nE))
    end
    return eL2, eLInf
end

function test_SIAC_l2proSINEWAVE(nE0, nR, pmin, pmax, showplots = false)
    fig = axes = nothing
    if showplots
        fig = plt.figure(1,frameon=false)
        axes = fig.subplots(nrows=2, ncols=pmax-pmin+1)
    end

    for o = 0:pmax-pmin
        p = pmin + o
        eL2   = zeros(nR + 1, 2)
        eLInf = zeros(nR + 1, 2)
        elmts = zeros(Int64,nR + 1)
        for k = 1:nR+1
            elmts[k] = 2 ^(k - 1) * nE0
            eL2[k,:],eLInf[k,:] = filter_l2_projection(p,elmts[k],[axes[2 * o + 1] axes[2 * o + 2] ] )
        end
    	@printf("\n\n=========================== RESULTS FOR p =%d ===============================\n\n",p)
        @printf("N\t ||e||_0\t ORDER\t ||e*||_0\t ORDER\t ||e||_inf\t ORDER\t ||e*||_inf\t ORDER\n")
        print("-----------------------------------------------------------------------------------\n")
        for k in 1:nR+1
            if k == 1
                @printf("%5i\t %.2e\t NA\t %.2e\t NA\t %.2e\t NA\t %.2e\t NA\n"  , elmts[k], eL2[k,1],eL2[k,2],eLInf[k,1],eLInf[k,2])
            else
                r2 = log10.(eL2[k-1,:]./eL2[k,:])./log10(elmts[k] / elmts[k-1])
                ri = log10.(eLInf[k-1,:]./eLInf[k,:])./log10(elmts[k] / elmts[k-1])
                @printf("%5i\t %.2e\t %.2f\t %.2e\t %.2f\t %.2e\t %.2f\t %.2e\t %.2f\t\n",
                         elmts[k], eL2[k,1],r2[1], eL2[k,2],r2[2], eLInf[k,1],ri[1],eLInf[k,2],ri[2])
            end
        end
		print("-----------------------------------------------------------------------------------\n")
        if showplots
            names = ["Projection" "Filtered"]
            for j = 1:length(axes)
                axes[j].spines["top"].set_visible(false)
                axes[j].spines["right"].set_visible(false)
                axes[j].set_yscale("log")
				if j <= 2
					xshift = axes[j].get_xlim() .- 0.3 * (pmax - pmin + 1)
                    yshift = axes[j].get_ylim()
                    axes[j].text(xshift[1],sqrt(yshift[1] * yshift[2]),names[j],family="serif", size=15)
                end
				if j %2 == 1
					axes[j].set_title("P = "*string(p))
				end
            end
            axes[end-1].legend(frameon=false,bbox_to_anchor=(1.05, 1),loc="upper left")
        end
    end
    show(block=false)
    sleep(20)
    close()
end


test_SIAC_l2proSINEWAVE(10,4,1,1,true)
