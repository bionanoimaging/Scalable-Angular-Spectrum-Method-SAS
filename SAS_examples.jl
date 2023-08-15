### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 4f0ea44a-5475-11ed-3979-6d7d4c1a8ce1
using FFTW, NDTools, Interpolations, IndexFunArrays, Colors, ImageShow, ImageIO, FourierTools, Plots, Interpolations, PlutoUI, ColorSchemes, TestImages

# ╔═╡ b1486e8d-0b5e-4d17-ac74-1f277596a660
using Random;

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

# ╔═╡ 5d639e0e-bc45-472b-8cb6-b78678671047
function damp_edge_outside_cpx(arr, damp_range)
	return complex.(damp_edge_outside(real.(arr), damp_range), damp_edge_outside(imag.(arr), damp_range))
end

# ╔═╡ b06b3800-df52-4e8b-94b0-6627ab3e3f82
""" 
	apply_parabolic_phase!(field_tuple)

applies the parabolic phase stored as the named parameters `dq` and `curvature` in the second member of the field tuple to the field stored in the first member of the field tuple.
"""
function apply_parabolic_phase!(field_tuple)
	sz = size(field_tuple[1])
	q_y = similar(field_tuple[1], sz[1])
	# fftpos generates coordinates from -L/2 to L/2 but excluding the last 
	# final bit
	q_y .= fftpos(field_tuple[2].dq * sz[1], sz[1], CenterFT)
	q_x = q_y'
	field_tuple[1] .*= exp.(1im .* field_tuple[2].curvature .* (q_x .^ 2 .+ q_y .^2))
end

# ╔═╡ 37b1e328-8c6f-4099-ab1f-89f1f09975c8
"""
	resample_to(arr, ref; damp_range=0.03, apply_parabolic=true)

Resamples the input amplitude to the sampling of the reference and extracts a common region in both fields. If `apply_parabolic` is true, the parabolic phase, stored in the first field will be applied to the resampled field. 
"""
function resample_to(arr, ref; damp_range=0.1, apply_parabolic=true)
	damp_arr = damp_edge_outside_cpx(arr[1], (damp_range, damp_range));
	# myzoom below is only used for the parabolic phase, which is why we use the
	# wanted zoom and not the actual zoom (nz ./ size(damp_arr));
	myzoom = arr[2].sampling ./ ref[2].sampling
	nz = round.(Int, size(damp_arr) .* myzoom);
	resarr = fftshift(resample(ifftshift(damp_arr), nz)) * sqrt(prod(size(damp_arr)) / prod(nz));
	sz = min.(size(ref[1]), nz);
	res_arr = select_region(resarr, new_size=sz);
	res_ref = select_region(ref[1], new_size=sz)
	if (apply_parabolic)
		if (hasproperty(arr[2],:dq))
			apply_parabolic_phase!((res_arr, (;dq=arr[2].dq/myzoom[1], curvature=arr[2].curvature)));
		end
		if (hasproperty(ref[2],:dq))
			apply_parabolic_phase!((res_ref, ref[2]));
		end
	end
	return res_arr, res_ref
end

# ╔═╡ 1f0cdb0e-440a-4bbc-861c-1fd8293fb8c3
"""
	amp_difference(arr, ref; damp_range=0.1)

Subtracts the intensities derived from two amplitudes with potentially different sampling and sizes. 
"""
function amp_difference(arr, ref; damp_range=0.03, apply_parabolic=true)
	res_arr, res_ref = resample_to(arr, ref; damp_range=damp_range, apply_parabolic=apply_parabolic)
	return res_arr .- res_ref
end

# ╔═╡ 5e0cdbd0-41a7-4cdc-9c91-06efe20a4769
""" 
	compare(arr, ref; damp_range=0.03)

The comparison is based on resampling `arr` and cutting both arrays to smaller of both sizes.
"""
function compare(arr, ref; damp_range=0.03, apply_parabolic=true)
	arr, ref = resample_to(arr, ref; damp_range=damp_range, apply_parabolic=apply_parabolic)
	return sum(abs2.(arr .- ref)) / sum(abs2.(ref))
end

# ╔═╡ 7038515e-17a8-4207-9551-4c2ef00a85b1
begin  # just a small test for the compare function to see if it works
	qq = rand(ComplexF64, 100,100);
	tt = (qq,(;sampling=0.123,));
	ss = (.-qq, (;sampling=0.123,));
	compare(ss,tt, apply_parabolic=false)
end

# ╔═╡ 45e9dad4-8f26-45c6-b58e-93d634881f60
md"# Angular Spectrum of Plane Waves"

# ╔═╡ 2c6d0d9b-9617-40d3-859d-4c5de8cafbd7
md"# Fresnel Propagation"

# ╔═╡ 004097d8-1906-4151-a4f3-4be7f7a71434
md"# Scaled Angular Spectrum"

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
M=4

# ╔═╡ e05c6882-81f9-4784-ab7e-7a9a8d296b6d
"""
    _propagation_variables(field, z, λ, L)

Internal method to create variables we need for propagation such as frequencies in Fourier space, etc.
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

# ╔═╡ adfcc771-e092-4dd3-8ff9-9a940c1c29a3
"""
	angular_spectrum(field, z, λ, L; pad_factor = 2, apply_bandlimit=true, bandlimit_border=(0.9, 1),  use_czt_zoom = false)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the angular spectrum method of plane waves (AS) by the propagation distance `z`.

* `pad_factor` allows to defined how much zero-padding relative to the field to propagate is used.
* `apply_bandlimt` defines whether high-angle rays (undersampled phases in Fourier space) are suppressed
* `bandlimit_border` provides two values defining the start and end of the band-limit filter to apply.
* `use_czt_zoom` applies a zoom into the region, such that the band_limit is placed at the XY-border. This changes the pixelsize

"""
function angular_spectrum(field::Matrix{T}, z, λ, L; 
					pad_factor=2, apply_bandlimit=true, bandlimit_border=(0.9, 1), 
					use_czt_zoom=false) where T
	
	@assert size(field, 1) == size(field, 2) "Restricted to quadratic fields."
	# we need to apply padding to prevent circular convolution
	L_new = pad_factor .* L
	
	# applies zero padding
	field_new = select_region(field, new_size=size(field) .* pad_factor)
	
	ft_field = 0;
	if (use_czt_zoom)
		# zoom: Fourier space zoom factor 
		# Δu =   zoom / L_new # Fourier pixel size in reciprocal meters
		# limit in Fourier-space frequencies, reciprocal meters:
		# sz = size(field_new,1)
		# Δu*sz != u_limit = 1 / (sqrt((2 * Δu * z)^2 + 1) * λ) 
		# λ^2 (Δu*sz)^2 ((2 Δu z)^2 +1) = 1
		#  (Δu)^2 ((2 Δu)^2 +1/z^2) = 1/(λ^2 sz^2 z^2)
		# (Δu)^2
		# zoom = L_new / ()
		# according to Yz et al. (2012), the with the samples in Freq Domain M
		# and real size S = L_new/zoom:
		# S = 2sqrt(2)zMλ / sqrt(-M^2 λ^2 + sqrt(M^4λ^4 + 64 z^2M^2λ^2))
		# zoom = L_new sqrt(-M^2 λ^2 + sqrt(M^4λ^4 + 64 z^2M^2λ^2)) / (2sqrt(2)zMλ)
		M = size(field_new, 1)
		zoom = L_new * sqrt(-M^2*λ^2 + sqrt(M^4*λ^4 + 64*z^2*M^2*λ^2)) / (2*sqrt(2)*z*M*λ)
		field_new .*= zoom
		ft_field = ifftshift(czt(field_new, (1/zoom, 1/zoom))) #
		L /= zoom;
		L_new /= zoom;
	else
		ft_field = fft(ifftshift(field_new))
	end
	
	# helpful propagation variables
	(; k, f_x, f_y) = _propagation_variables(field_new, z, λ, L_new)
	
	# transfer function kernel of angular spectrum
	H = exp.(1im .* k .* z .* sqrt.(0im .+ 1 .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)))
	
	if (apply_bandlimit)
		# bandlimit according to Matsushima
		# as addition we introduce a smooth bandlimit with a Hann window
		# and fuzzy logic 
		Δu =   1 / L_new
		u_limit = 1 / (sqrt((2 * Δu * z)^2 + 1) * λ)
		smooth_f(x, α, β) = hann(scale(x, α, β))	
		
		# bandlimit filter
		# smoothing at 0.8 is arbitrary but works well
		W = .*(smooth_f.(abs2.(f_y) ./ u_limit^2 .+ abs2.(f_x) * λ^2, bandlimit_border[1], bandlimit_border[2]),
			 smooth_f.(abs2.(f_x) ./ u_limit^2 .+ abs2.(f_y) * λ^2, bandlimit_border[1], bandlimit_border[2]))

		# apply band-limit
		H .*= W;
	end
	# propagate field
	field_out = fftshift(ifft(ft_field .* H ))
	# take center part because of circular convolution
	field_out_cropped = select_region(field_out, new_size=size(field))
	
	# return final field and some other variables
	return field_out_cropped, (; L=L, sampling=L/size(field,1))
end

# ╔═╡ 2177f522-9ccb-4b96-8bd5-92718f0d5cc6
"""
	fresnel(field, z, λ, L; skip_final_phase=true)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the fresnel method of plane waves (AS) by the propagation distance `z`.

"""
	function fresnel(field::Matrix{T}, z, λ, L; skip_final_phase=true) where T
	@assert size(field, 1) == size(field, 2) "Restricted to quadratic fields."
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
	
	# calculate phases of Fresnel
	H₁ = exp.(1im .* k ./ (2 .* z) .* (x .^ 2 .+ y .^ 2))

	# skips multiplication of final phase
	field_out = exp.(1im .* k .* z) .* fftshift(fft(ifftshift(field_new) .* H₁))
	
	# fix scaling
	field_out .*= 1 / (1im * T(sqrt(length(field_out)))) 
	
	# transfer function kernel of angular spectrum
	field_tuple = (field_out, (; L=Q, sampling=Q/size(field_out,1), dq =dq, curvature = k ./ (2 .* z)))

	if (!skip_final_phase)
		apply_parabolic_phase!(field_tuple)
	end
	return field_tuple
end

# ╔═╡ 4db3a990-4e5d-4fe7-89cc-4823d1b5b592
"""
	scalable_angular_spectrum(field, z, λ, L; skip_final_phase=true, apply_bandlimit = true)

Returns the the electrical field with physical length `L` and wavelength `λ` propagated with the Scaled Angular Spectrum (SAS) of plane waves (AS) by the propagation distance `z`.


"""
function scalable_angular_spectrum(ψ₀::Matrix{T}, z, λ, L ; 
								 pad_factor=2, skip_final_phase=true,  set_pad_zero=false, 
								bandlimit_border=(0.9, 1), apply_bandlimit = true) where {T} 
	@assert size(ψ₀, 1) == size(ψ₀, 2) "Restricted to quadratic fields."
	
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

	# ΔH is the core part of Fresnel and AS
	H_AS = sqrt.(0im .+ 1 .- abs2.(f_x .* λ) .- abs2.(f_y .* λ)) 
	H_Fr = 1 .- abs2.(f_x .* λ) / 2 .- abs2.(f_y .* λ) / 2 
	
	if (apply_bandlimit)
		# smooth window function
		smooth_f(x, α, β) = hann(scale(x, α, β))
		# find boundary for soft hann
		ineq_x = fftshift(cx[1, :].^2 .* (1 .+ tx[1, :].^2) ./ tx[1, :].^2 .+ cy[1, :].^2)
		limits = find_width_window(ineq_x, bandlimit_border)
	
		# bandlimit filter for precompensation
		W = .*(smooth_f.(cx.^2 .* (1 .+ tx.^2) ./ tx.^2 .+ cy.^2, limits...),
		 		smooth_f.(cy.^2 .* (1 .+ ty.^2) ./ ty.^2 .+ cx.^2, limits...))
		# take the difference here, key part of the ScaledAS
		ΔH = W .* exp.(1im .* k .* z .* (H_AS .- H_Fr)) 
	else
		W = one.(H_AS);
		# take the difference here, key part of the ScaledAS
		ΔH = exp.(1im .* k .* z .* (H_AS .- H_Fr)) 
	end
	
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
		
	# calculate phases of Fresnel
	H₁ = exp.(1im .* k ./ (2 .* z) .* (x .^ 2 .+ y .^ 2))

	# skip final phase because often undersampled
	ψ_p_final = exp.(1im .* k .* z) .* fftshift(fft(H₁ .* ψ_precomp))
	
	# fix absolute scaling of field
	ψ_p_final .*= 1 / (1im * T(sqrt(length(ψ_precomp)))) 
	# unpad/crop/extract center
	ψ_final = select_region(ψ_p_final, new_size=size(ψ₀))
	
	field_tuple = (ψ_final, (;L=L * M, W, sampling=(L*M)/size(ψ_final,1), dq =λ * z / L_new, curvature = k ./ (2 .* z)))

	
	if (!skip_final_phase)
		apply_parabolic_phase!(field_tuple)
	end
	
	return field_tuple
end

# ╔═╡ 6fa374ce-6953-443a-94a0-9859237fe345
z_circ = M / N /λ * L^2 * 2

# ╔═╡ c7194950-26ff-4972-81be-1fabf1ba9dcf
md"# First Example: Circular Aperture

In the first example, one straight beam and one oblique beam are passing through a round aperture.

The Fresnel number is $(round((D_circ / 2 * L / N)^2 / z_circ / λ, digits=3))
"

# ╔═╡ 6a977f39-626a-441a-816a-66f4b8e0c64c
z_circ / L

# ╔═╡ 0cd5c3e8-39ca-40be-8fef-17faf7738b45
simshow(U_circ)

# ╔═╡ 77f6528c-cf26-465e-a5bd-7bd336e1b4bc
@time as_circ = angular_spectrum(select_region(U_circ, new_size=round.(Int, size(U_circ) .* M)), pad_factor=6, z_circ, λ, L * M)

# ╔═╡ 3c7bb832-d0db-4f39-913f-6f36d931f0bc
@time as_czt_circ = angular_spectrum(select_region(U_circ, new_size=round.(Int, size(U_circ) .* M)), pad_factor=2, z_circ, λ, M*L, apply_bandlimit=true, use_czt_zoom=true);

# ╔═╡ 6af0bc99-4245-44f8-bc45-405f9e56b513
@time sas_circ = scalable_angular_spectrum(U_circ, z_circ, λ, L, apply_bandlimit=true);

# ╔═╡ 3afb5c51-889f-43e4-9ef8-94bf63a31b6f
@time sas_circ_nb = scalable_angular_spectrum(U_circ, z_circ, λ, L, apply_bandlimit=false); # to see the benefit of the band limit

# ╔═╡ 16a40756-2b97-4965-8627-831ecc6de4c8
@time sft_fr_circ = fresnel(select_region(U_circ, M=2), z_circ, λ, 2 * L, skip_final_phase=true)

# ╔═╡ dd434bfd-c14d-4417-922a-01a573c44143
sft_fr_circ_cut = select_region(sft_fr_circ[1], M=0.5);

# ╔═╡ 1e6e8a5a-b757-44a5-9b23-70eb711da252
compare(as_czt_circ, as_circ, damp_range=0.1)*100 # quantitative comparison

# ╔═╡ 2e97955f-bed0-41ab-a1c6-d309c5b2b565
compare(sft_fr_circ, as_circ)*100 # quantitative comparison

# ╔═╡ 5c401fe7-3029-44f7-87cc-5c4f3d6ec58d
compare(sas_circ, as_circ, damp_range=0.1)*100 # quantitative comparison

# ╔═╡ c413248c-d082-4580-844f-916597852eb0
compare(sas_circ_nb, as_circ, damp_range=0.1)*100 # quantitative comparison

# ╔═╡ 3524374c-97f0-4cdd-88cd-7ffbdb52834c
simshow(abs2.(as_circ[1]), γ=0.13, cmap=:inferno)

# ╔═╡ b95302c7-0385-46ac-8f53-2e6cf7cecea9
simshow(abs2.(sft_fr_circ_cut), γ=0.13, cmap=:inferno)

# ╔═╡ c4f2b545-cd1d-4ae2-bccb-7a89119ae7df
simshow(abs2.(sas_circ[1]), γ=0.13, cmap=:inferno)

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
md"# Second Example: Square Aperture


The Fresnel number is $(round((D_box)^2 / z_box / λ, digits=3))
"

# ╔═╡ e4bb5e06-0b89-4c27-885f-0d13da6d2ff0
simshow(U_box)

# ╔═╡ 9d78321e-6586-4c31-bec7-279d23c79841
@time as_box = angular_spectrum(select_region(U_box, new_size=round.(Int, size(U_box) .* M_box)), z_box, λ, L_box * M_box);

# ╔═╡ 5fde2e0b-4bdb-4bcd-8985-9a0b54cbbf95
@time sft_fr_box = fresnel(select_region(U_box, M=2), z_box, λ, L_box, skip_final_phase=true);

# ╔═╡ dc0ae388-c96d-4e9b-bd1b-0c752ddfa237
@time sft_fr_box_displ = select_region(sft_fr_box[1], M=1//2);

# ╔═╡ b3e31f75-5216-47b5-85b3-026a0321c0a8
@time sas_box = scalable_angular_spectrum(U_box, z_box, λ, L_box, skip_final_phase=true);

# ╔═╡ 4f9aab5a-8c2b-4424-97be-4d4b0ba07b3b
compare(sas_box, as_box)*100

# ╔═╡ f7874387-fed8-41a9-9b66-3c0847e485b6
compare(sft_fr_box, as_box)*100

# ╔═╡ d128d0ec-61bd-46a2-a915-e42220cd09cc
simshow(abs2.(as_box[1]), γ=0.13, cmap=:inferno)

# ╔═╡ ac013a5b-9225-4ce2-9e6a-7d83c94f5aa6
simshow(abs2.(sft_fr_box_displ), γ=0.13, cmap=:inferno)

# ╔═╡ 9c46ad96-96ac-4d40-bfec-d146451f1130
simshow(abs2.(sas_box[1]), γ=0.13, cmap=:inferno)

# ╔═╡ 2f79966d-86a5-4066-a84e-a128c93247e8
md"# Third Example: 5x5 Beamsplitter-Hologram


Reproduce the example from Asoubar et al. JOS-A 31, 591, 2014 (Fig. 5 & Table 1)
A hologram generates 5x5 beamlets at 10mm distance. The publication does not state the specifics of the hologram used. The table (Table 1) states the following values:
- AS (SPW) propagation: 17521 x 17521, 0%
- Fresnel: 462 x 462,  145%
- Semianalytical SPW propagation (Taylor): 2053 x 2053,  0.006%
- Semianalytical SPW propagation (Avoort): 1252 x 1252,  0.006%
presumably the field to propagate has 462 x 462 pixels with the following parameters:
- λ = 532 nm
- L = 2 * 277.3 µm
Since Asoubar et al. do not disclose the exact pattern to propagate ('diffractive beam splitter, which consists of 16 discrete phase levels'), we calculated such phase levels by the help of an IFTA algorithm.
"

# ╔═╡ d0841cb8-3b2a-4242-8769-fe9e2bca4915
λh = 532e-9; Lh = 0.837 * 2*277.3e-6; Nh = 462; # the factor 0.837 is to make the result size of sas directly agree to the destination field size

# ╔═╡ 0e84ef74-2f4c-42d9-bfc1-92c5d82c459a
Δxh = Lh/Nh # sampling in the source plane

# ╔═╡ de1ef254-c6bc-4c31-ac03-2ee1cf57ed18
yh = fftpos(Lh, Nh, NDTools.CenterFT);

# ╔═╡ b4b5e6b2-a80f-4dd4-b12e-e991a9bad15a
sigma = (100e-6/2)/sqrt(2)/Δxh; # sigma for the 100µm (2w0) width of the Gaussian

# ╔═╡ 7330bc0f-d4f5-47a3-b8da-4df91e04f987
illuh = gaussian((Nh,Nh), sigma=sigma);

# ╔═╡ 047b310a-01f7-45c3-af35-1a5fb1b1a2ad
z_h = 10e-3; # 1cm propagation distance

# ╔═╡ 4fcd98e4-6980-4716-bd32-ade190e07f20
Lh / Nh / λh

# ╔═╡ d5369ebe-ac4a-4a70-9ebb-7ac189e85a55
z_h / Lh

# ╔═╡ d080f246-0c37-45d3-9ea4-b766d908c797
maxkrel_h = sin(atan(0.5/10)); # 0.5mm spot distance @ 10mm propagation;

# ╔═╡ 794d4546-6b82-4d67-9ea8-265b520cfffe
function discretize(amp, numlevels=16)
	 return exp.(1im .* 2pi.*round.((angle.(amp).+2pi)./2pi.*numlevels)./numlevels);
end

# ╔═╡ 295f3bf2-6353-4927-b418-45c989100c20
function replace_abs(amp, myabs)
	return exp.(1im.*angle.(amp)) .* myabs;
end

# ╔═╡ 92e06b9d-2f6a-4dd2-a6c7-3ea1787655ea
function get_hologram(λ, y, max_krel)
	res = zeros(ComplexF64, (size(y,1), size(y,1)));
	Random.seed!(42);
	for n = -2:2
		for m = -2:2
			res += exp(1im * 2π *rand()) .* exp.(1im .* 2π ./ λ .* (y .* (max_krel*m).+y' .* (max_krel*n)));
		end
	end
	return discretize(res);
end

# ╔═╡ 18a82e0b-4f69-42f1-b784-96be541173f1
function local_normalization(ftamp, ftreldist, bs=8)
	avg_int = 0;
	new_ft = ftamp .* 0;
	N = 0;
	for n = -2:2
		for m = -2:2
			mid = size(ftamp).÷2 .+1;
			k_start = mid .+ round.(Int, size(ftamp) .* ftreldist .* [m,n])
			roi = @view ftamp[k_start[1]-bs:k_start[1]+bs, k_start[2]-bs:k_start[2]+bs]
			sum_int = sum(abs2.(roi));
			avg_int += sum_int;
			roi_new = @view new_ft[k_start[1]-bs:k_start[1]+bs, k_start[2]-bs:k_start[2]+bs]
			roi_new .= roi ./ sqrt(sum_int);
			N+=1;
		end
	end
	avg_int = avg_int/N;
	new_ft .*= sqrt(avg_int);
	return new_ft;
end

# ╔═╡ ec92354b-2cba-49a4-998a-7c6925733200
function IFTA_hologram(illuh, λ, y, max_krel, steps=50)
	holo = get_hologram(λ, y, max_krel)
	Δx = yh[2]-yh[1];
	ftreldist = maxkrel_h *Δx/λ;
	sz = size(illuh);
	mid_region = [sz[1].÷4:(3*sz[1]).÷4, sz[2].÷4:(3*sz[2]).÷4]
	# obtain a normalized version and define 80% of it as the "goal"
	ftamp = ft(illuh .* holo);
	goal_abs_amp = abs.(local_normalization(ftamp, ftreldist)) .* 0.7;
	goal_abs_amp_view =@view goal_abs_amp[mid_region...]
	for n=1:steps
		ftamp = ft(illuh .* holo);
		mid_view = @view ftamp[mid_region...]
		mid_view .= replace_abs(mid_view, goal_abs_amp_view)
		holo = ift(ftamp);
		holo = discretize(holo);
	end
	local_normalization(ft(illuh .* holo), ftreldist);
	return holo;
end

# ╔═╡ bd264ae6-4dd3-493e-bd5f-b42444b620dd
holo = IFTA_hologram(illuh, λh, yh, maxkrel_h, 150);

# ╔═╡ af9c6f62-706b-46df-a68a-8cd4e552e7c4
U_h = illuh .* holo;

# ╔═╡ bd2457e6-6fca-41cc-b67c-7358912348d7
simshow(illuh)

# ╔═╡ 61c6b13a-ff42-4eba-97d7-8820e4c59ac5
simshow(U_h, γ=1)

# ╔═╡ 00540988-e8e7-41cf-92b1-1afc91ea9ac8
simshow(abs2.(ft(U_h)), γ=1)

# ╔═╡ 35d43276-2ddd-4bed-902c-8d8e705e766a
local_normalization(ft(U_h),  maxkrel_h *Δxh/λh);

# ╔═╡ 3d7762e1-0c57-471d-86ef-86dcc282a486
M_h = (1.3277e-3*2)/Lh  # about 6-fold zero padding, needed to make the AS propagated final field agree to the desired area.

# ╔═╡ 67cc47fa-61d0-4dd1-a4cc-1ad7ad33c687
U_h_padded = select_region(U_h, new_size=round.(Int, size(U_h) .* M_h));

# ╔═╡ 3141c07d-8cb2-4b72-9d89-87a4458af7ee
@time as_h = angular_spectrum(U_h_padded, z_h, λh, Lh*M_h, pad_factor=2, apply_bandlimit=false);

# ╔═╡ 95bfa1d7-ce87-4fa0-a153-d76021946d44
int_as = abs2.(as_h[1]);

# ╔═╡ dbbc6bcd-cc3d-4e8f-a987-0aab81729f13
Lh

# ╔═╡ 05391390-f6ed-4123-a4fe-3e529feaf544
@time sas_h = scalable_angular_spectrum(U_h, z_h, λh, Lh, bandlimit_border=(0.95, 1));

# ╔═╡ 2b51977e-9257-4c8c-86bf-541f0b3799cd
int_sas = abs2.(sas_h[1]);

# ╔═╡ 47fa52ab-ce09-4120-a5d6-214f91ae18e6
@time sas_hnb = scalable_angular_spectrum(U_h, z_h, λh, Lh, apply_bandlimit=false);

# ╔═╡ 9accf8da-5b5a-4b08-b853-d0cec2259bcd
int_sas_nb = abs2.(sas_hnb[1]);

# ╔═╡ 70a2a5a1-7070-4ab4-8426-fb9e7218c042
# NumPix_ASPW = M_h*Lh / Δxh * pad_factor

# ╔═╡ 31d5feec-df3e-4e97-b824-eab4cda3b93b
int_h = abs2.(as_h[1]);

# ╔═╡ 3c56caa8-89ff-4c8d-98d6-614dc1cb43ed
@time as_hb = angular_spectrum(U_h, z_h, λh, Lh, pad_factor=1, apply_bandlimit=true);

# ╔═╡ 7f8dcada-f408-4c4e-923d-387de4eef636
int_hb = abs2.(as_hb[1]);

# ╔═╡ ea32264a-2482-4909-8ed3-3a7f8b54cdda
@time as_hb2 = angular_spectrum(U_h, z_h, λh, Lh, pad_factor=2, apply_bandlimit=true);

# ╔═╡ 7a070377-98ee-4656-92b7-85743a356871
int_hb2 = abs2.(as_hb2[1]);

# ╔═╡ 199d9441-2e64-4ff6-84ac-9f06ca69dc9c
@time fr_h = fresnel(U_h, z_h, λh, Lh, skip_final_phase=false)

# ╔═╡ 6f540081-52e5-4ae1-8c56-719b45bd07e0
fr_h[2].L*1e3

# ╔═╡ e73facc3-d24d-49fc-8536-94dbc5705bc7
simshow(int_sas, γ=1, cmap=:gray)

# ╔═╡ 366cfdfa-0611-4a3f-9eba-28b7adf04f30
L_dest = sas_h[2].L*1e3 # field width in mm in the destination plane

# ╔═╡ 881b95cc-1da9-4d61-a594-55b320b14994
L_dest_Fr = fr_h[2].L * 1e3

# ╔═╡ 281f7278-9525-4896-b91c-87451c6b4992
simshow(int_as, γ=1, cmap=:gray)

# ╔═╡ abac7af2-f619-4554-9184-489ffbbec3b4
L_dest_as = Lh * 1e3 # field width in mm in the destination plane

# ╔═╡ 6e9b6bdd-bf91-473f-ab6d-c75ed2e82fa1
md"# Comparison of Quality metrics
with the norm according to Asoubar et al.

No padding with AS propagation yields:
"

# ╔═╡ e8a6256e-b083-4f5c-b58b-5712f2f91b6d
md"Matsushima suppression (no padding):"

# ╔═╡ 2e5f40b6-92ea-4bd5-b7bc-5b9a2d67e483
compare(as_hb, as_h)*100 # AS no padding (!)

# ╔═╡ c15c53c5-8cb6-41e9-a6cd-08ab08fa07a7
compare(as_h, as_hb)*100 # just to compare the other way round

# ╔═╡ e141037c-afae-41de-802f-d4aaadaada32
md"Matsushima suppression (2x padding):"

# ╔═╡ 21038f44-2f99-4ea5-abf2-81f22f04ef54
compare(as_hb2, as_h)*100 # AS only 2x padding, but with Matsushima-bandlimit

# ╔═╡ 31c0eb54-f3ad-46f2-be1d-c42e531a4a14
md"Fresnel propagation (no padding):"

# ╔═╡ f7e2ddff-4561-430d-8fa9-9b2b4384dc57
compare(fr_h, as_h)*100 # In the paper they claim 145% difference.

# ╔═╡ e449bf56-3d69-4c8e-867d-54fff56755bc
size(fr_h[1],1) # to double-check whether the size is OK

# ╔═╡ b89b8461-f97d-42e2-bee1-44b8979b8198
z_h # propagation distance of 1cm, to double-check

# ╔═╡ ea70c9cf-f212-4176-a0aa-6c88b9a2ee9e
as_h[2].L*1e3 # fieldsize in mm

# ╔═╡ 61d11c3b-0a2c-44b2-982a-4bd9cd03334c
as_h[2].sampling

# ╔═╡ 5fe99558-6a4a-49a2-9d92-8ad78da3306d
simshow(abs2.(fr_h[1]))

# ╔═╡ 0b252803-b3fe-4745-9e13-f2e493db1ba8
simshow(as_h[1])

# ╔═╡ 1a8c6f5c-d9c8-4d09-b88c-5995317c5fa1
size(as_h[1])

# ╔═╡ 99267895-ba7d-4efb-b994-647d3d6457eb
simshow(resample_to(sas_h, as_h)[1])

# ╔═╡ 56028dcb-edcc-4abb-b378-d38758767dc2
simshow(damp_edge_outside_cpx(sas_h[1], (0.2, 0.2))) # looks fine

# ╔═╡ b7722ea3-c47d-4127-8cfc-840834be920d
md"SAS propagation (2x padding):"

# ╔═╡ d219c3d5-6e68-434c-a2b6-e07111238e75
# if you change damp_range just a little the metrics varies by orders of magnitude
compare(sas_h, as_h)*100 # SAS, 2x padding (924x924), resampled

# ╔═╡ e540c671-7361-40e4-83e3-c5bbc3f175bb
compare(sas_hnb, as_h)*100 # SAS, 2x padding (924x924), resampled

# ╔═╡ 20a00ae2-fa54-4f56-ad1b-2c204fc85d2b
compare(as_h, sas_hnb)*100 # SAS, 2x padding (924x924), resampled

# ╔═╡ 397c1577-4b79-43a9-8302-2818fbbbff83
toshow = amp_difference(sas_h, as_h);

# ╔═╡ 8e5166da-961b-404d-bff4-b0db703ff5a4
maximum(abs.(toshow))

# ╔═╡ 774dcce4-deb8-44f3-a9e6-d3ab7346c48d
simshow(toshow, γ=0.2, cmap=:gray, set_zero=false)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FourierTools = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
IndexFunArrays = "613c443e-d742-454e-bfc6-1d7f8dd76566"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
NDTools = "98581153-e998-4eef-8d0d-5ec2c052313d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
ColorSchemes = "~3.21.0"
Colors = "~0.12.10"
FFTW = "~1.7.1"
FourierTools = "~0.4.2"
ImageIO = "~0.6.6"
ImageShow = "~0.3.7"
IndexFunArrays = "~0.2.6"
Interpolations = "~0.14.7"
NDTools = "~0.5.2"
Plots = "~1.38.16"
PlutoUI = "~0.7.51"
TestImages = "~1.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "a7db9b1b205997e256188c843ab57fdcc8159cdd"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractNFFTs]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "292e21e99dedb8621c15f185b8fdb4260bb3c429"
uuid = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
version = "0.8.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "4e88377ae7ebeaf29a047aa1ee40826e0b708a5d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.7.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FourierTools]]
deps = ["ChainRulesCore", "FFTW", "IndexFunArrays", "LinearAlgebra", "NDTools", "NFFT", "PaddedViews", "Reexport", "ShiftedArrays"]
git-tree-sha1 = "8967a9d259ab1c50e3b3abc6b77d3e3d829d2e6d"
uuid = "b18b359b-aebc-45ac-a139-9c0ccbb2871e"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2613d054b0e18a3dea99ca1594e9a3960e025da4"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.7"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7607ad4100c75908a79ff31fabb792cd37711d70"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+4"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndexFunArrays]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "e66a2aeb6d5814015004080e5203dfff44d2856f"
uuid = "613c443e-d742-454e-bfc6-1d7f8dd76566"
version = "0.2.6"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NDTools]]
deps = ["LinearAlgebra", "OffsetArrays", "PaddedViews", "Random", "Statistics"]
git-tree-sha1 = "10e35b25261dfd361045e16afa9db5b14a4c1184"
uuid = "98581153-e998-4eef-8d0d-5ec2c052313d"
version = "0.5.2"

[[deps.NFFT]]
deps = ["AbstractNFFTs", "BasicInterpolators", "Distributed", "FFTW", "FLoops", "LinearAlgebra", "Printf", "Random", "Reexport", "SnoopPrecompile", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "93a5f32dd6cf09456b0b81afcb8fc29f06535ffd"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.13.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "ceeef74797d961aee825aabf71446d6aba898acb"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.2"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "03492434a1bdde3026288939fc31b5660407b624"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.7.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "8621f5c499a8aa4aa970b1ae381aae0ef1576966"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.4"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "a66fb81baec325cf6ccafa243af573b031e87b00"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.77"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─2bd51f20-7abb-4d95-a56b-c2e058c2a1be
# ╟─15975d27-b575-4e76-94a7-02b8f218acb1
# ╠═4f0ea44a-5475-11ed-3979-6d7d4c1a8ce1
# ╠═45dabf95-ede9-46c5-896c-39945a2029e7
# ╠═83d8201f-6c96-4849-871b-99409abfc5f8
# ╟─d0e12818-286b-475d-b76b-da777073e72a
# ╟─b87f5371-13b0-4c73-91fb-8108a5a80a3e
# ╟─e53711d2-68ed-4712-82ba-c11bc14ffab3
# ╟─529a3df1-8a18-416e-9730-14eb10302fbb
# ╟─5d639e0e-bc45-472b-8cb6-b78678671047
# ╟─b06b3800-df52-4e8b-94b0-6627ab3e3f82
# ╟─37b1e328-8c6f-4099-ab1f-89f1f09975c8
# ╟─1f0cdb0e-440a-4bbc-861c-1fd8293fb8c3
# ╟─5e0cdbd0-41a7-4cdc-9c91-06efe20a4769
# ╟─7038515e-17a8-4207-9551-4c2ef00a85b1
# ╟─e05c6882-81f9-4784-ab7e-7a9a8d296b6d
# ╟─45e9dad4-8f26-45c6-b58e-93d634881f60
# ╠═adfcc771-e092-4dd3-8ff9-9a940c1c29a3
# ╟─2c6d0d9b-9617-40d3-859d-4c5de8cafbd7
# ╠═2177f522-9ccb-4b96-8bd5-92718f0d5cc6
# ╟─004097d8-1906-4151-a4f3-4be7f7a71434
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
# ╠═0cd5c3e8-39ca-40be-8fef-17faf7738b45
# ╠═77f6528c-cf26-465e-a5bd-7bd336e1b4bc
# ╠═3c7bb832-d0db-4f39-913f-6f36d931f0bc
# ╠═6af0bc99-4245-44f8-bc45-405f9e56b513
# ╠═dd434bfd-c14d-4417-922a-01a573c44143
# ╠═3afb5c51-889f-43e4-9ef8-94bf63a31b6f
# ╠═16a40756-2b97-4965-8627-831ecc6de4c8
# ╠═1e6e8a5a-b757-44a5-9b23-70eb711da252
# ╠═2e97955f-bed0-41ab-a1c6-d309c5b2b565
# ╠═5c401fe7-3029-44f7-87cc-5c4f3d6ec58d
# ╠═c413248c-d082-4580-844f-916597852eb0
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
# ╠═5fde2e0b-4bdb-4bcd-8985-9a0b54cbbf95
# ╠═dc0ae388-c96d-4e9b-bd1b-0c752ddfa237
# ╠═b3e31f75-5216-47b5-85b3-026a0321c0a8
# ╠═4f9aab5a-8c2b-4424-97be-4d4b0ba07b3b
# ╠═f7874387-fed8-41a9-9b66-3c0847e485b6
# ╠═d128d0ec-61bd-46a2-a915-e42220cd09cc
# ╠═ac013a5b-9225-4ce2-9e6a-7d83c94f5aa6
# ╠═9c46ad96-96ac-4d40-bfec-d146451f1130
# ╟─2f79966d-86a5-4066-a84e-a128c93247e8
# ╠═d0841cb8-3b2a-4242-8769-fe9e2bca4915
# ╠═0e84ef74-2f4c-42d9-bfc1-92c5d82c459a
# ╠═de1ef254-c6bc-4c31-ac03-2ee1cf57ed18
# ╠═b4b5e6b2-a80f-4dd4-b12e-e991a9bad15a
# ╠═7330bc0f-d4f5-47a3-b8da-4df91e04f987
# ╠═047b310a-01f7-45c3-af35-1a5fb1b1a2ad
# ╠═4fcd98e4-6980-4716-bd32-ade190e07f20
# ╠═d5369ebe-ac4a-4a70-9ebb-7ac189e85a55
# ╠═d080f246-0c37-45d3-9ea4-b766d908c797
# ╠═b1486e8d-0b5e-4d17-ac74-1f277596a660
# ╠═794d4546-6b82-4d67-9ea8-265b520cfffe
# ╠═295f3bf2-6353-4927-b418-45c989100c20
# ╠═92e06b9d-2f6a-4dd2-a6c7-3ea1787655ea
# ╠═18a82e0b-4f69-42f1-b784-96be541173f1
# ╠═ec92354b-2cba-49a4-998a-7c6925733200
# ╠═bd264ae6-4dd3-493e-bd5f-b42444b620dd
# ╠═af9c6f62-706b-46df-a68a-8cd4e552e7c4
# ╠═bd2457e6-6fca-41cc-b67c-7358912348d7
# ╠═61c6b13a-ff42-4eba-97d7-8820e4c59ac5
# ╠═00540988-e8e7-41cf-92b1-1afc91ea9ac8
# ╠═35d43276-2ddd-4bed-902c-8d8e705e766a
# ╠═3d7762e1-0c57-471d-86ef-86dcc282a486
# ╠═67cc47fa-61d0-4dd1-a4cc-1ad7ad33c687
# ╠═3141c07d-8cb2-4b72-9d89-87a4458af7ee
# ╠═95bfa1d7-ce87-4fa0-a153-d76021946d44
# ╠═dbbc6bcd-cc3d-4e8f-a987-0aab81729f13
# ╠═05391390-f6ed-4123-a4fe-3e529feaf544
# ╠═2b51977e-9257-4c8c-86bf-541f0b3799cd
# ╠═47fa52ab-ce09-4120-a5d6-214f91ae18e6
# ╠═9accf8da-5b5a-4b08-b853-d0cec2259bcd
# ╠═70a2a5a1-7070-4ab4-8426-fb9e7218c042
# ╠═31d5feec-df3e-4e97-b824-eab4cda3b93b
# ╠═3c56caa8-89ff-4c8d-98d6-614dc1cb43ed
# ╠═7f8dcada-f408-4c4e-923d-387de4eef636
# ╠═ea32264a-2482-4909-8ed3-3a7f8b54cdda
# ╠═7a070377-98ee-4656-92b7-85743a356871
# ╠═199d9441-2e64-4ff6-84ac-9f06ca69dc9c
# ╠═6f540081-52e5-4ae1-8c56-719b45bd07e0
# ╠═e73facc3-d24d-49fc-8536-94dbc5705bc7
# ╠═366cfdfa-0611-4a3f-9eba-28b7adf04f30
# ╠═881b95cc-1da9-4d61-a594-55b320b14994
# ╠═281f7278-9525-4896-b91c-87451c6b4992
# ╠═abac7af2-f619-4554-9184-489ffbbec3b4
# ╟─6e9b6bdd-bf91-473f-ab6d-c75ed2e82fa1
# ╟─e8a6256e-b083-4f5c-b58b-5712f2f91b6d
# ╠═2e5f40b6-92ea-4bd5-b7bc-5b9a2d67e483
# ╠═c15c53c5-8cb6-41e9-a6cd-08ab08fa07a7
# ╟─e141037c-afae-41de-802f-d4aaadaada32
# ╠═21038f44-2f99-4ea5-abf2-81f22f04ef54
# ╟─31c0eb54-f3ad-46f2-be1d-c42e531a4a14
# ╠═f7e2ddff-4561-430d-8fa9-9b2b4384dc57
# ╠═e449bf56-3d69-4c8e-867d-54fff56755bc
# ╠═b89b8461-f97d-42e2-bee1-44b8979b8198
# ╠═ea70c9cf-f212-4176-a0aa-6c88b9a2ee9e
# ╠═61d11c3b-0a2c-44b2-982a-4bd9cd03334c
# ╠═5fe99558-6a4a-49a2-9d92-8ad78da3306d
# ╠═0b252803-b3fe-4745-9e13-f2e493db1ba8
# ╠═1a8c6f5c-d9c8-4d09-b88c-5995317c5fa1
# ╠═99267895-ba7d-4efb-b994-647d3d6457eb
# ╠═56028dcb-edcc-4abb-b378-d38758767dc2
# ╠═b7722ea3-c47d-4127-8cfc-840834be920d
# ╠═d219c3d5-6e68-434c-a2b6-e07111238e75
# ╠═e540c671-7361-40e4-83e3-c5bbc3f175bb
# ╠═20a00ae2-fa54-4f56-ad1b-2c204fc85d2b
# ╠═397c1577-4b79-43a9-8302-2818fbbbff83
# ╠═8e5166da-961b-404d-bff4-b0db703ff5a4
# ╠═774dcce4-deb8-44f3-a9e6-d3ab7346c48d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
