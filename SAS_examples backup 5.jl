### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 4f0ea44a-5475-11ed-3979-6d7d4c1a8ce1
using FFTW, NDTools, Interpolations, IndexFunArrays, Colors, ImageShow, ImageIO, FourierTools, Plots, Interpolations, PlutoUI, ImageView, ColorSchemes, TestImages

# ╔═╡ 2bd51f20-7abb-4d95-a56b-c2e058c2a1be
md"# Scaled Angular Spectrum
Here we present the Julia implementation of the Scaled Angular Spectrum method and show the examples from the article.
"

# ╔═╡ 15975d27-b575-4e76-94a7-02b8f218acb1
md"## Load Packages"

# ╔═╡ 45dabf95-ede9-46c5-896c-39945a2029e7
begin
	FFTW.forget_wisdom()
	FFTW.set_num_threads(4)
end

# ╔═╡ 83d8201f-6c96-4849-871b-99409abfc5f8
TableOfContents()

# ╔═╡ d0e12818-286b-475d-b76b-da777073e72a
md"## Some Utility Functions"

# ╔═╡ b87f5371-13b0-4c73-91fb-8108a5a80a3e
hann(x) = sinpi(x/2)^2

# ╔═╡ e53711d2-68ed-4712-82ba-c11bc14ffab3
scale(x, α, β) = clamp(α ≥ β ? (x > β ? zero(x) : one(x)) : 1 - (x-α)/(β-α), 0, 1)

# ╔═╡ 529a3df1-8a18-416e-9730-14eb10302fbb
function find_width_window(ineq_1D::AbstractVector, bandlimit_border)
	bs = ineq_1D .≤ 1
	ind_x_first = findfirst(bs)
	ind_x_last = findlast(bs)

	if isnothing(ind_x_first) || isnothing(ind_x_last)
		return (1,1)
	end
	
	diff_b = round(Int, 0.5 * (1 - bandlimit_border[1]) * (Tuple(ind_x_last)[1] - Tuple(ind_x_first)[1]))
	diff_e = round(Int, 0.5 * (1 - bandlimit_border[2]) * (Tuple(ind_x_last)[1] - 
	Tuple(ind_x_first)[1]))
	
	ineq_v_b = ineq_1D[ind_x_first + diff_b]
	ineq_v_e = ineq_1D[ind_x_first + diff_e]

	return ineq_v_b, ineq_v_e
end

# ╔═╡ e05c6882-81f9-4784-ab7e-7a9a8d296b6d
"""
    _propagation_variables(field, z, λ, L)

Internal method to create variables we need for propagation such as frequencies in Fourier space, etc..
"""
function _propagation_variables(field::AbstractArray{T, M}, z, λ, L) where {T, M} 
	@assert size(field, 1) == size(field, 2) "Quadratic fields only working currently"
	
	# wave number
	k = T(2π) / λ
	# number of samples
	N = size(field, 1)
	# sample spacing
	dx = L / N 
	# frequency spacing
	df = 1 / L 
	# total size in frequency space
	Lf = N * df
	
	# frequencies centered around first entry 
	# 1D vectors each
	f_y = similar(field, real(eltype(field)), (N,))
	f_y .= fftfreq(N, Lf)
	f_x = collect(f_y')
	
	# y and x positions in real space
	#y = ifftshift(range(-L/2, L/2, length=N))
	y = similar(field, real(eltype(field)), (N,))
	y .= ifftshift(fftpos(L, N, CenterFT))
	x = collect(y')
	
	return (; k, dx, df, f_x, f_y, x, y)
end

# ╔═╡ 45e9dad4-8f26-45c6-b58e-93d634881f60
md"# Angular Spectrum of Plane Waves"

# ╔═╡ adfcc771-e092-4dd3-8ff9-9a940c1c29a3
"""
	angular_spectrum(field, z, λ, L)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum method of plane waves (AS) by the propagation distance `z`.
"""
function angular_spectrum(field::Matrix{T}, z, λ, L; pad_factor = 2) where T
	@assert size(field, 1) == size(field, 2) "Restricted to auadratic fields."
	# we need to apply padding to prevent circular convolution
	L_new = pad_factor .* L
	
	# applies zero padding
	field_new = select_region(field, new_size=size(field) .* pad_factor)
	
	# helpful propagation variables
	(; k, f_x, f_y) = _propagation_variables(field_new, z, λ, L_new)
	
	# transfer function kernel of angular spectrum
	H = exp.(1im .* k .* z .* sqrt.(0im .+ 1 .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	
	# bandlimit according to Matsushima
	# as addition we introduce a smooth bandlimit with a Hann window
	# and fuzzy logic 
	Δu =   1 / L_new
	u_limit = 1 / (sqrt((2 * Δu * z)^2 + 1) * λ)
	smooth_f(x, α, β) = hann(scale(x, α, β))	
	
	# bandlimit filter
	# smoothing at 0.8 is arbitrary but works well
	W = .*(smooth_f.(abs2.(f_y) ./ u_limit^2 .+ abs2.(f_x) * λ^2, 0.8, 1),
		 smooth_f.(abs2.(f_x) ./ u_limit^2 .+ abs2.(f_y) * λ^2, 0.8, 1))
	
	# propagate field
	field_out = fftshift(ifft(fft(ifftshift(field_new)) .* H .* W))
	# take center part because of circular convolution
	field_out_cropped = select_region(field_out, new_size=size(field))
	
	# return final field and some other variables
	return field_out_cropped, (; )
end

# ╔═╡ 2c6d0d9b-9617-40d3-859d-4c5de8cafbd7
md"# Fresnel Propagation"

# ╔═╡ 2177f522-9ccb-4b96-8bd5-92718f0d5cc6
"""
	fresnel(field, z, λ, L; skip_final_phase=true)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the fresnel method of plane waves (AS) by the propagation distance `z`.


"""
	function fresnel(field::Matrix{T}, z, λ, L; skip_final_phase=true) where T
	@assert size(field, 1) == size(field, 2) "Restricted to auadratic fields."
	# we need to apply padding to prevent circular convolution
	pad_factor = 1
	L_new = pad_factor .* L
	# applies zero padding
	field_new = pad_factor != 1 ?
			select_region(field, new_size=size(field) .* pad_factor) :
			field
	
	# helpful propagation variables
	(; k, f_x, f_y, x, y) = _propagation_variables(field_new, z, λ, L_new)
	
	
	N = size(field_new, 1)
	# new sample coordinates
	M = λ * z * N / L_new^2
	dq = λ * z / L_new
	Q = dq * N
	
	q_y = similar(field, N)
	q_y .= ifftshift(fftpos(M * L_new, N, CenterFT))
	q_x = q_y'
	
	# calculate phases of Fresnel
	H₁ = exp.(1im .* k ./ (2 .* z) .* (x .^ 2 .+ y .^ 2))

	# skips multiplication of final phase
	if skip_final_phase
		field_out = fftshift(fft(ifftshift(field_new) .* H₁))
	else
		H₂ = (exp.(1im .* k .* z) .*
			 exp.(1im .* k ./ (2 .* z) .* (q_x .^ 2 .+ q_y .^2)))
		field_out = fftshift(fft(ifftshift(field_new) .* H₁) .* H₂)
	end
	
	# fix scaling
	field_out .*= 1 / (1im * T(sqrt(length(field_out)))) 
	
	# transfer function kernel of angular spectrum
	return field_out, (; L=Q)
end

# ╔═╡ 004097d8-1906-4151-a4f3-4be7f7a71434
md"# Scaled Angular Spectrum"

# ╔═╡ fdb237d3-5c00-463c-9671-3de7ee3e2bcc


# ╔═╡ 4db3a990-4e5d-4fe7-89cc-4823d1b5b592
"""
	scalable_angular_spectrum(field, z, λ, L; skip_final_phase=true)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the Scaled Angular Spectrum (SAS) of plane waves (AS) by the propagation distance `z`.
"""
function scalable_angular_spectrum(ψ₀::Matrix{T}, z, λ, L ; 
								 pad_factor=2, skip_final_phase=true,  set_pad_zero=false, bandlimit_soft_px=20,
								bandlimit_border=(0.8, 1)) where {T} 
	@assert bandlimit_soft_px ≥ 0 "bandlimit_soft_px must be ≥ 0"
	@assert size(ψ₀, 1) == size(ψ₀, 2) "Restricted to auadratic fields."
	
	
	N = size(ψ₀, 1)
	z_limit = (- 4 * L * sqrt(8*L^2 / N^2 + λ^2) * sqrt(L^2 * inv(8 * L^2 + N^2 * λ^2)) / (λ * (-1+2 * sqrt(2) * sqrt(L^2 * inv(8 * L^2 + N^2 * λ^2)))))
	
	# vignetting limit
	z > z_limit &&  @warn "Propagated field might be affected by vignetting"
	L_new = pad_factor * L
	
	# applies zero padding
	ψ_p = select_region(ψ₀, new_size=size(ψ₀) .* pad_factor)
	k, dx, df, f_x, f_y, x, y = _propagation_variables(ψ_p, z, λ, L_new)  
	M = λ * z * N / L^2 / 2
	
	# calculate anti_aliasing_filter for precompensation
	cx = λ .* f_x 
	cy = λ .* f_y 
	tx = L_new / 2 / z .+ abs.(λ .* f_x)
	ty = L_new / 2 / z .+ abs.(λ .* f_y)
	
	# smooth window function
	smooth_f(x, α, β) = hann(scale(x, α, β))
	# find boundary for soft hann
	ineq_x = fftshift(cx[1, :].^2 .* (1 .+ tx[1, :].^2) ./ tx[1, :].^2 .+ cy[1, :].^2)
	limits = find_width_window(ineq_x, bandlimit_border)

	# bandlimit filter for precompensation
	W = .*(smooth_f.(cx.^2 .* (1 .+ tx.^2) ./ tx.^2 .+ cy.^2, limits...),
	 		smooth_f.(cy.^2 .* (1 .+ ty.^2) ./ ty.^2 .+ cx.^2, limits...))
	
	# ΔH is the core part of Fresnel and AS
	H_AS = sqrt.(0im .+ 1 .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)) 
	H_Fr = 1 .- abs2.(f_x .* λ) / 2 .- abs2.(f_y .* λ) / 2 
	# take the difference here, key part of the ScaledAS
	ΔH = W .* exp.(1im .* k .* z .* (H_AS .- H_Fr)) 

	# apply precompensation
	ψ_precomp = ifft(fft(ifftshift(ψ_p)) .* ΔH)
	
	# we can set the padding region to zero
	# but quite often there is meaningful signal
	# not used, just for debugging
	if set_pad_zero
		ψ_precomp = select_region(fftshift(ψ_precomp), new_size=size(field))
		ψ_precomp = select_region(ψ_precomp, new_size=pad_factor .* size(ψ₀))
		ψ_precomp = ifftshift(ψ_precomp)
	end
	
	# new sample coordinates
	dq = λ * z / L_new
	Q = dq * N * pad_factor
	q_y = similar(ψ_p, pad_factor * N)
	# fftpos generates coordinates from -L/2 to L/2 but excluding the last 
	# final bit
	q_y .= ifftshift(fftpos(dq * pad_factor * N, pad_factor * N, CenterFT))
	q_x = q_y'
	
	# calculate phases of Fresnel
	H₁ = exp.(1im .* k ./ (2 .* z) .* (x .^ 2 .+ y .^ 2))

	# skip final phase because often undersampled
	if skip_final_phase
		ψ_p_final = fftshift(fft(H₁ .* ψ_precomp))
	else
		H₂ = (exp.(1im .* k .* z) .*
			 exp.(1im .* k ./ (2 .* z) .* (q_x .^ 2 .+ q_y .^2)))
		ψ_p_final = fftshift(H₂ .* fft(H₁ .* ψ_precomp))
	end
	
	# fix absolute scaling of field
	ψ_p_final .*= 1 / (1im * T(sqrt(length(ψ_precomp)))) 
	# unpad/crop/extract center
	ψ_final = select_region(ψ_p_final, new_size=size(ψ₀))
	
	return ψ_final, (;Q, L=L * M, W)
end

# ╔═╡ fd94ba72-5130-40e8-884c-37899b2f2fa7
λ = 500e-9

# ╔═╡ 0c62447e-cc20-4e68-a645-367dd823b507
L = 128e-6 / 2

# ╔═╡ 8cc751a6-aa3a-4095-97a3-256ba37d3faa
N = 512

# ╔═╡ 76fc433a-fc15-4660-b7ad-436c4d756488
L / N / λ

# ╔═╡ a53ce9e8-1f86-40ee-949c-378cf486af1b
y = fftpos(L, N, NDTools.CenterFT)

# ╔═╡ d58452cc-d4b7-4d6f-9c39-fd8329291cdd
D_circ = N / 8

# ╔═╡ 5611f23e-8513-4c6d-b2d5-b092bdff21ed
U_circ = ComplexF64.(rr((N, N)) .< D_circ / 2) .* exp.(1im .* 2π ./ λ .* y .* sind(45)) .+ ComplexF64.(rr((N, N)) .< D_circ / 2) .* exp.(1im .* 2π ./ λ .* y' .* sind(-45));

# ╔═╡ 694b3ac0-51e2-46b4-a5ce-8b1a93d6a368
M = 4

# ╔═╡ 6fa374ce-6953-443a-94a0-9859237fe345
z_circ = M / N /λ * L^2 * 2

# ╔═╡ c7194950-26ff-4972-81be-1fabf1ba9dcf
md"# First Example: Circular

In the first example, one straight beam and one oblique beam are passing through a round aperture.

The Fresnel number is $(round((D_circ / 2 * L / N)^2 / z_circ / λ, digits=3))
"

# ╔═╡ 6a977f39-626a-441a-816a-66f4b8e0c64c
z_circ / L

# ╔═╡ 764349e1-3b12-412b-bcdd-ba1bb64bc391
z_circ

# ╔═╡ 0cd5c3e8-39ca-40be-8fef-17faf7738b45
simshow(U_circ)

# ╔═╡ 77f6528c-cf26-465e-a5bd-7bd336e1b4bc
@time as_circ = angular_spectrum(select_region(U_circ, new_size=round.(Int, size(U_circ) .* M)), z_circ, λ, L * M)

# ╔═╡ dd434bfd-c14d-4417-922a-01a573c44143
@time sft_fr_circ = select_region(fresnel(select_region(U_circ, M=2), z_circ, λ, 2 * L, skip_final_phase=true)[1], M=0.5);

# ╔═╡ 6af0bc99-4245-44f8-bc45-405f9e56b513
@time sas_circ = scalable_angular_spectrum(U_circ, z_circ, λ, L, bandlimit_border=(0.9, 1));

# ╔═╡ 3524374c-97f0-4cdd-88cd-7ffbdb52834c
simshow(abs2.(as_circ[1]), γ=0.13, cmap=:inferno)

# ╔═╡ b95302c7-0385-46ac-8f53-2e6cf7cecea9
simshow(abs2.(sft_fr_circ), γ=0.13, cmap=:inferno)

# ╔═╡ c4f2b545-cd1d-4ae2-bccb-7a89119ae7df
simshow(abs2.(sas_circ[1]), γ=.13, cmap=:inferno)

# ╔═╡ d623e68d-8cfd-4df8-af30-396097ddc6aa
L_box = 128e-6;

# ╔═╡ 81c307a0-82d4-4514-8d28-12e12defcea2
N_box = 512;

# ╔═╡ 01f39e27-8e6a-4056-b496-d6bdf955120f
y_box = fftpos(L_box, N_box, NDTools.CenterFT);

# ╔═╡ 930bc90e-a55f-4674-a6f8-246efa183520
x_box = y_box';

# ╔═╡ 22812caa-acc6-4a50-bdb0-d43b153c9c9a
D_box = L_box / 16

# ╔═╡ 840f8832-ee38-4da5-b722-e9022fca3076
U_box = (x_box.^2 .<= (D_box / 2).^2) .* (y_box.^2 .<= (D_box / 2).^2) .* (exp.(1im .* 2π ./ λ .* y_box' .* sind(20)));

# ╔═╡ af91c034-2f43-4786-aef7-a7bce45ab38e
M_box = 8;

# ╔═╡ 7b13f72d-6e5d-440b-b080-1301a1560acc
z_box = M_box / N_box / λ * L_box^2 * 2

# ╔═╡ 1815437a-332c-4bc1-9b72-b75cd4b8b653
md"# Second Example: Quadratic


The Fresnel number is $(round((D_box)^2 / z_box / λ, digits=3))
"

# ╔═╡ e4bb5e06-0b89-4c27-885f-0d13da6d2ff0
simshow(U_box)

# ╔═╡ 9d78321e-6586-4c31-bec7-279d23c79841
@time as_box = angular_spectrum(select_region(U_box, new_size=round.(Int, size(U_box) .* M_box)), z_box, λ, L_box * M_box);

# ╔═╡ dc0ae388-c96d-4e9b-bd1b-0c752ddfa237
@time sft_fr_box = select_region(fresnel(select_region(U_box, M=2), z_box, λ, L_box, skip_final_phase=true)[1], M=1//2);

# ╔═╡ b3e31f75-5216-47b5-85b3-026a0321c0a8
@time sas_box = scalable_angular_spectrum(U_box, z_box, λ, L_box, bandlimit_border=(0.8, 1.0), skip_final_phase=true);

# ╔═╡ d128d0ec-61bd-46a2-a915-e42220cd09cc
simshow(abs2.(as_box[1]), γ=0.13, cmap=:inferno)

# ╔═╡ ac013a5b-9225-4ce2-9e6a-7d83c94f5aa6
simshow(abs2.(sft_fr_box), γ=0.13, cmap=:inferno)

# ╔═╡ 9c46ad96-96ac-4d40-bfec-d146451f1130
simshow(abs2.(sas_box[1]), γ=0.13, cmap=:inferno)

# ╔═╡ 2f79966d-86a5-4066-a84e-a128c93247e8
md"# Third Example: Hologram


Reproduce the example from Aoubar et al. JOS-A 31, 591 (Fig. 5)
A hologram generates 5x5 beamlets at 10mm distance.
"

# ╔═╡ d0841cb8-3b2a-4242-8769-fe9e2bca4915
λh = 532e-9; Lh = 2*277.3e-6; Nh = 462;

# ╔═╡ de1ef254-c6bc-4c31-ac03-2ee1cf57ed18
yh = fftpos(Lh, Nh, NDTools.CenterFT);

# ╔═╡ 7330bc0f-d4f5-47a3-b8da-4df91e04f987
illuh = gaussian((Nh,Nh), sigma=20.0);

# ╔═╡ 047b310a-01f7-45c3-af35-1a5fb1b1a2ad
z_h = 10e-3;

# ╔═╡ 4fcd98e4-6980-4716-bd32-ade190e07f20
Lh / Nh / λh

# ╔═╡ d5369ebe-ac4a-4a70-9ebb-7ac189e85a55
z_h / Lh

# ╔═╡ a90bfca7-2c3e-4c92-97ee-1cc18f2f9692
U_h = illuh.*(exp.(1im .* 2π ./ λh .* yh .* sind(45)*0.1) .+ exp.(1im .* 2π ./ λh .* yh' .* sind(-45)*0.1));

# ╔═╡ 0cf14312-d9fe-4b30-a434-086811a38ddc


# ╔═╡ 61c6b13a-ff42-4eba-97d7-8820e4c59ac5
simshow(U_h)

# ╔═╡ 05391390-f6ed-4123-a4fe-3e529feaf544
@time sas_h = scalable_angular_spectrum(U_h, z_h, λh, Lh, bandlimit_border=(0.9, 1));

# ╔═╡ e73facc3-d24d-49fc-8536-94dbc5705bc7
simshow(abs2.(sas_h[1]), γ=0.13, cmap=:gray)

# ╔═╡ 366cfdfa-0611-4a3f-9eba-28b7adf04f30
sas_h[2].L*1e3 # field width in mm

# ╔═╡ 27386598-bf41-4bff-8ad8-ea40536c7d02


# ╔═╡ Cell order:
# ╟─2bd51f20-7abb-4d95-a56b-c2e058c2a1be
# ╟─15975d27-b575-4e76-94a7-02b8f218acb1
# ╠═4f0ea44a-5475-11ed-3979-6d7d4c1a8ce1
# ╠═45dabf95-ede9-46c5-896c-39945a2029e7
# ╠═83d8201f-6c96-4849-871b-99409abfc5f8
# ╟─d0e12818-286b-475d-b76b-da777073e72a
# ╠═b87f5371-13b0-4c73-91fb-8108a5a80a3e
# ╠═e53711d2-68ed-4712-82ba-c11bc14ffab3
# ╠═529a3df1-8a18-416e-9730-14eb10302fbb
# ╠═e05c6882-81f9-4784-ab7e-7a9a8d296b6d
# ╟─45e9dad4-8f26-45c6-b58e-93d634881f60
# ╠═adfcc771-e092-4dd3-8ff9-9a940c1c29a3
# ╟─2c6d0d9b-9617-40d3-859d-4c5de8cafbd7
# ╠═2177f522-9ccb-4b96-8bd5-92718f0d5cc6
# ╟─004097d8-1906-4151-a4f3-4be7f7a71434
# ╠═fdb237d3-5c00-463c-9671-3de7ee3e2bcc
# ╠═4db3a990-4e5d-4fe7-89cc-4823d1b5b592
# ╟─c7194950-26ff-4972-81be-1fabf1ba9dcf
# ╠═fd94ba72-5130-40e8-884c-37899b2f2fa7
# ╠═0c62447e-cc20-4e68-a645-367dd823b507
# ╠═76fc433a-fc15-4660-b7ad-436c4d756488
# ╠═6a977f39-626a-441a-816a-66f4b8e0c64c
# ╠═8cc751a6-aa3a-4095-97a3-256ba37d3faa
# ╠═a53ce9e8-1f86-40ee-949c-378cf486af1b
# ╠═d58452cc-d4b7-4d6f-9c39-fd8329291cdd
# ╠═5611f23e-8513-4c6d-b2d5-b092bdff21ed
# ╠═694b3ac0-51e2-46b4-a5ce-8b1a93d6a368
# ╠═6fa374ce-6953-443a-94a0-9859237fe345
# ╠═764349e1-3b12-412b-bcdd-ba1bb64bc391
# ╠═0cd5c3e8-39ca-40be-8fef-17faf7738b45
# ╠═77f6528c-cf26-465e-a5bd-7bd336e1b4bc
# ╠═dd434bfd-c14d-4417-922a-01a573c44143
# ╠═6af0bc99-4245-44f8-bc45-405f9e56b513
# ╠═3524374c-97f0-4cdd-88cd-7ffbdb52834c
# ╠═b95302c7-0385-46ac-8f53-2e6cf7cecea9
# ╠═c4f2b545-cd1d-4ae2-bccb-7a89119ae7df
# ╟─1815437a-332c-4bc1-9b72-b75cd4b8b653
# ╠═d623e68d-8cfd-4df8-af30-396097ddc6aa
# ╠═81c307a0-82d4-4514-8d28-12e12defcea2
# ╠═01f39e27-8e6a-4056-b496-d6bdf955120f
# ╠═930bc90e-a55f-4674-a6f8-246efa183520
# ╠═22812caa-acc6-4a50-bdb0-d43b153c9c9a
# ╠═840f8832-ee38-4da5-b722-e9022fca3076
# ╠═af91c034-2f43-4786-aef7-a7bce45ab38e
# ╠═7b13f72d-6e5d-440b-b080-1301a1560acc
# ╠═e4bb5e06-0b89-4c27-885f-0d13da6d2ff0
# ╠═9d78321e-6586-4c31-bec7-279d23c79841
# ╠═dc0ae388-c96d-4e9b-bd1b-0c752ddfa237
# ╠═b3e31f75-5216-47b5-85b3-026a0321c0a8
# ╠═d128d0ec-61bd-46a2-a915-e42220cd09cc
# ╠═ac013a5b-9225-4ce2-9e6a-7d83c94f5aa6
# ╠═9c46ad96-96ac-4d40-bfec-d146451f1130
# ╟─2f79966d-86a5-4066-a84e-a128c93247e8
# ╠═d0841cb8-3b2a-4242-8769-fe9e2bca4915
# ╠═de1ef254-c6bc-4c31-ac03-2ee1cf57ed18
# ╠═7330bc0f-d4f5-47a3-b8da-4df91e04f987
# ╠═047b310a-01f7-45c3-af35-1a5fb1b1a2ad
# ╠═4fcd98e4-6980-4716-bd32-ade190e07f20
# ╠═d5369ebe-ac4a-4a70-9ebb-7ac189e85a55
# ╠═a90bfca7-2c3e-4c92-97ee-1cc18f2f9692
# ╠═0cf14312-d9fe-4b30-a434-086811a38ddc
# ╠═61c6b13a-ff42-4eba-97d7-8820e4c59ac5
# ╠═05391390-f6ed-4123-a4fe-3e529feaf544
# ╠═e73facc3-d24d-49fc-8536-94dbc5705bc7
# ╠═366cfdfa-0611-4a3f-9eba-28b7adf04f30
# ╠═27386598-bf41-4bff-8ad8-ea40536c7d02
