using Plots
# using FieldPropagation

Rs = 0.5001:0.001:15.0 # SourcePixelsize / λ 
Rsf = 0.01:0.01:15.0
# R = L / (λ N) = SourcePixelsize / λ
z_L(R)=4/(1/R-2*sqrt(2)/sqrt(1+8*R^2)) # z_L is really z/L
function z_L2(R)
    λf = inv(sqrt(16*R^2+2))
    return inv(λf/sqrt(1-2*λf^2)-λf)
    #return 1-2*λf^2
end
function z_L3(R)
    inv(inv(4*R)-inv(sqrt(16*R^2+2)))
end
function z_L4(R)
    4*R*sqrt(16*R^2+2)/(sqrt(16*R^2+2)-4*R)
end

zeroax = 0.0001 .*Rsf
min_q1 = 
maxax = 1e10 .*Rsf

"""
    z_L_Fresnel(pitch_over_lamba)

minimum propagation distance to warrant the Fresnel limit for 2x padding
"""
function z_L_Fresnel(pitch_over_lamba)
    2*pitch_over_lamba
end

plot(Rsf,z_L.(Rsf), fillrange=z_L_Fresnel.(Rsf), fillalpha=0.2, xlabel="pixel pitch / λ", yaxis=:log, ylabel="z / L", label="SAS Vignetting Limit",legend=:bottomright, color=:green)
# plot(Rs,z_L.(Rs), fillrange=zeroax, fillalpha=0.2, label=nothing,color=:green)
plot!(Rsf, z_L_Fresnel.(Rsf), fillrange=zeroax, fillalpha=0.2, label="Minimum distance (M=1)", color=:yellow)
ylims!((0.01,1e6))


"""
    z_L_AS(pitch_over_lamba)

maximum propagation distance for 2x padding and Angular Spectrum (AS) Propagation
"""
function z_L_AS(pitch_over_lamba)
    2*sqrt.(4*pitch_over_lamba.^2 .- 1)
end
# plot!(Rs, z_L_AS.(Rs), label="AS maximum distance", color=:green, linestyle=:dash)

N = 256.0
vline!([0.5], xticks=[0.5,(2:2:14)...],seriestype = :vline, label="λ/2 pixel pitch", color=:blue)

# Angular spectrum limit where a frequency entirely disappears,
# assuming 100% zero padding in source and destination planes
R_as(R) = sqrt(4*R^2-1) # R = L/(lambda N)

function other_x!(myxlims=(0.0, 10.0), func=(x)->1/(2x), ifunc=(y)->1/(2y), xtick=[0.01,0.05,0.1,0.2,0.3,1.0]; numdigits=2, name = "NAₚᵢₓₑₗ")
    xlims!(myxlims)
    DX = myxlims[2]-myxlims[1]
    tick_pos = (ifunc.(xtick).-myxlims[1])./DX
    newpos = (x)->func(myxlims[1]+x*DX)
    # formatter is given a linear relative x-position and computes the label string
    myformatter = x->string(round(Int, 10^numdigits*newpos(x))/10^numdigits)
    display(plot!(twiny(), xtick=tick_pos, xaxis=(formatter=myformatter),xlabel=name))
end

# a distance is also related to an NA via the beam in from the optical axis to the side
z_L(R)=4/(1/R-2*sqrt(2)/sqrt(1+8*R^2)) # z_L is really z/L

# M = round(z_L(myR) ./ myR / 2 *100)/100
# zL = round(z_L.(myR)*100)/100
# M = (z/L) / (pix/λ) / 2
# NA = sin(atan(ML/z))
# z/L = M/tan(asin(NA))
function z_L_from_na(na)
    @show na
    Ms = z_L.(Rsf) ./ Rsf ./ 2
    zLs = z_L.(Rsf)
    nas = sin.(atan.(Ms ./ zLs))
    pos = findfirst(na.>nas)
    return Ms[pos]/tan(asin(na))
end

function na_from_z_L(zL)
    @show zL
    Ms = z_L.(Rsf) ./ Rsf ./ 2
    zLs = z_L.(Rsf)
    nas = sin.(atan.(Ms ./ zLs))
    pos = findfirst(zL.<zLs)
    return nas[pos]
end

function R_from_M(M)
    # 1/abs2(M) = (1-2*sqrt(2)*R/sqrt(1+8*R^2))^2/4
    # 4/abs2(M) = 1 + 4*2*R^2/(1+8*R^2) - 4*sqrt(2)*R/sqrt(1+8*R^2)
    # 1/abs2(M) - 1/4 =  2*R^2/(1+8*R^2) - sqrt(2)*R/sqrt(1+8*R^2)
    # 1/abs2(M) - 1/4 =  (2*R^2- sqrt(2)*R*sqrt(1+8*R^2))/(1+8*R^2) 
    # (1+8*R^2)/abs2(M) - (1+8*R^2)/4 = 2*R^2- sqrt(2)*R*sqrt(1+8*R^2)
    # 1/abs2(M)+8*R^2/abs2(M) - 1/4+8*R^2/4 = 2*R^2- sqrt(2)*R*sqrt(1+8*R^2)
    @show M
    Ms = z_L.(Rsf) ./ Rsf ./ 2
    # zLs = z_L.(Rsf)
    pos = findfirst(M.<Ms)
    return Rsf[pos]
end

function show_Ms()
    for M in (2.0, 3.0, 5.0, 10.9, 20, 100, 200, 500, 1000, 2000)
        myR = R_from_M(M)
        # M = round(z_L(myR) ./ myR / 2 *100)/100
        zL = round(z_L.(myR)*100)/100
        NA = round(na_from_z_L(zL)*100)/100
        # NA = na_from_z_L(zL)
        display(annotate!(myR-0.08, z_L.(myR), text("∘           M=$(M), NAₘ=$(NA)", :blue, :left, 10)))
        # display(annotate!(myR-0.08, z_L.(myR), text("∘ M=$(M), z/L=$(zL)", :green, :left, 10)))
    end
end

show_Ms()

function plot_RM(R, M, col=:green, sym="x", toleft=0.09; move_down=nothing, thetext=nothing)
    zL = M * 2 * R
    if isnothing(thetext)
        thetext = "M=$(M)";
    end
    if isnothing(move_down)
        thetext = "$(sym)   $(thetext)"
        display(annotate!(R-toleft, zL, text(thetext, col, :left, 8)))
    else
        display(annotate!(R-toleft, zL, text("$(sym)", col, :left, 8)))
        thetext = "     $(thetext)"
        display(annotate!(R-toleft, zL-move_down, text(thetext, col, :left, 8)))
    end
end

# Add a green dot for R=0.5, M=5
plot_RM(0.5, 5, :green, move_down=1.0)
# Add a purple dot for R=0.5, M=15
plot_RM(0.5, 15, :purple)

# The parameters of Asoubar et al. JOS-A 31, 591 (Fig. 5 & Table 1) 
# λ_h = 532 nm
# L_h = 837 * 2 * 277.3 µm = 
# z_h = 10e-3; # 1cm propagation distance
# pixel_pitch / λ =  1µm / 0.532µm = 1.8797
# z/L = 10e3 / 464200 = 21.54
plot_RM(1.8797, 21.54, :black, thetext="Asoubar et al.")

# Fig. 6: M∘=4, R = (64µm / 512) / 0.5µm = 0.25
plot_RM(0.25, 4, :black, "∘", 0.08)
# Fig. 7: M_⧆=8, R = (128µm / 512) / 0.5µm = 0.5
plot_RM(0.5, 8, :black, "⧆", 0.08)

other_x!((0,14))


# savefig("propagation_distance_sas_3ax.pdf")

########## The code below is currently unused as a second Y-axis is debatable

function other_y!(myylims=(0.1, 1E6), func=(na)->na_from_z_L(na), ifunc=(na)->z_L_from_na(na), 
                  ytick=[0.04,0.05,0.1,0.2,0.3,0.5,0.8]; numdigits=2, name = "NAₚᵣₒₚ")
    ylims!(myylims)
    # DY = myylims[2]-myylims[1]
    logfac = log(myylims[2]/myylims[1])
    # calculate all relative tick positions
    tick_pos = log.(ifunc.(ytick)/myylims[1])./logfac
    # NA label in dependence of the linear y-position
    lin_to_log(y) = myylims[1]*exp(y*logfac)
    newpos = (y)->func(lin_to_log(y)) # myylims[1]+y*DY
    # formatter is given a linear relative y-position and computes the label string
    myformatter = y->string(round(Int, 10^numdigits*newpos(y))/10^numdigits)
    display(plot!(twinx(), ytick=tick_pos, yaxis=(formatter=myformatter), ylabel=name))
end
myylims=(0.1, 1E6)
ylims!(myylims)
other_y!()

# equation von Felix:
λ = 0.5566e-6
N = 256
# R = pix/lambda = L/(N λ)
# L = R * (λ N)
zf(L,λ,N)=4*L*sqrt(λ^2+8*L^2/N^2)*sqrt(L^2/(8L^2+λ^2*N^2))/λ/(1-2*sqrt(2)*sqrt(L^2/(8L^2+λ^2*N^2)))
plot!(Rs,zf.(Rs.*λ.*N,λ,N) ./ (Rs.*λ.*N), xlabel="pixelsize / λ", yaxis=:log, ylabel="distance / L", label="SAS limit Felix")

zf(100.0,1.0,1000)
z_L(0.1) * 1000

########## compare equation 18 with the exact expression 24
# 18:  Lp/z = (f_max λ) ^3
# 24:  Lp/z = 2 * abs(f_max λ / sqrt(1-λf_max^2) - λf_max) 

eq18(f_max_λ) = f_max_λ ^3
eq24(f_max_λ) = 2 * abs(f_max_λ / sqrt(1-f_max_λ^2) - f_max_λ)

f_max_λ = 0.01:0.01:0.95

plot(f_max_λ, eq18.(f_max_λ), label="Eq. 18", xlabel="f_max λ", ylabel= "L / z")
plot!(f_max_λ, eq24.(f_max_λ), label="Eq. 24")

