#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragPos;

layout(location = 0) out vec4 outColor;

float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdSphere( vec3 p, float s ){
  return length(p)-s;
}

float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float map_the_world(in vec3 p)
{
	vec3 rep = vec3(10, 10, 10);
	vec3 q = mod(p+0.5*rep,rep)-0.5*rep;
    // float sphere_0 = distance_from_sphere(q, vec3(0.0), 1.0);
	// float sphere_0 = min(sdSphere(q, .5), sdTorus(q, vec2(1.0, .25)));
	float sphere_0 = sdSphere(q, 5);
	rep = .5*vec3(1, 1, 1);
	q = mod(p+0.5*rep,rep)-0.5*rep;
	sphere_0 = max(-sdRoundBox(q, .2*vec3(1.0, 1.0, 1.0), 0), sphere_0);

    return sphere_0; //sdTorus(q, vec2(1.0, .25));
}

vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map_the_world(p + small_step.xyy) - map_the_world(p - small_step.xyy);
    float gradient_y = map_the_world(p + small_step.yxy) - map_the_world(p - small_step.yxy);
    float gradient_z = map_the_world(p + small_step.yyx) - map_the_world(p - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

vec3 calcNormal(vec3 p ) // for function f(p)
{
    const float h = 0.0001; // replace by an appropriate value
    const vec2 k = vec2(1,-1);
    return normalize( k.xyy*map_the_world( p + k.xyy*h ) + 
                      k.yyx*map_the_world( p + k.yyx*h ) + 
                      k.yxy*map_the_world( p + k.yxy*h ) + 
                      k.xxx*map_the_world( p + k.xxx*h ) );
}

const vec3 light_position = vec3(5.0, 5.0, 0);

float shadowCast(in vec3 ro, in vec3 rd, float max_dist)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 1000;
    const float MINIMUM_HIT_DISTANCE = 0.001;
	float h = 0;
	float res = 1;
	float ph = 1e20;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        h = map_the_world(ro + total_distance_traveled * rd);
		

        if (h < MINIMUM_HIT_DISTANCE){
            return 0;
        }
		float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
		res = min( res, 128*d/max(0.0, total_distance_traveled-y) );

        if (total_distance_traveled > max_dist)
        {
            break;
        }
        total_distance_traveled += h;
    }
    return res;
}

vec3 ray_march(in vec3 ro, in vec3 rd)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 1000;
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        vec3 current_position = ro + total_distance_traveled * rd;

        float distance_to_closest = map_the_world(current_position);

        if (distance_to_closest < MINIMUM_HIT_DISTANCE)
        {
            vec3 normal = calcNormal(current_position);
            vec3 direction_to_light = normalize(light_position - current_position);

            float diffuse_intensity = max(0.0, dot(normal, direction_to_light));

			float alpha = 1.0 - exp( -total_distance_traveled*.1 );
            return mix((normal/2+1) * diffuse_intensity * shadowCast(current_position + .01*direction_to_light, direction_to_light, length(current_position - light_position)), vec3(0), alpha);
        }

        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            break;
        }
        total_distance_traveled += distance_to_closest;
    }
    return vec3(0);
}

void main()
{
    vec3 camera_position = vec3(5.0, 5.0, -5.0);
    vec3 ro = camera_position;
    vec3 rd = normalize(vec3(fragPos.x, fragPos.y, 1.0));

    vec3 shaded_color = ray_march(ro, rd);

    outColor = vec4(shaded_color, 1.0);
}