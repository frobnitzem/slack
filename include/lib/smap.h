/*
 *    smap version 1.0.0
 *
 *    ANSI C hash table for bytestrings.
 *
 *        Version history:
 *        1.0.0 - initial release
 *        2.0.0 - changed function prefix from strmap to sm to ensure
 *            ANSI C compatibility
 *        2.0.1 - improved documentation 
 *        1.0.0 - changes listed in map.c
 *
 *    smap.h
 *
 *    Copyright (c) 2009, 2011, 2013 Per Ola Kristensson.
 *    Copyright (c) 2013 David M. Rogers
 *
 *    Per Ola Kristensson <pok21@cam.ac.uk> 
 *    Inference Group, Department of Physics
 *    University of Cambridge
 *    Cavendish Laboratory
 *    JJ Thomson Avenue
 *    CB3 0HE Cambridge
 *    United Kingdom
 *
 *    This file is part of cmap.
 *
 *    cmap is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    cmap is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with cmap.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _SMAP_H
#define _SMAP_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <string.h>

typedef struct SMap SMap;

/*
 * This callback function is called once per key-value when iterating over
 * all keys associated to values.
 *
 * Parameters:
 *
 * key: A pointer to a bytestring. The string must not
 * be modified by the client.
 *
 * obj: A pointer to a client-specific object. This parameter may be
 * null.
 *
 * Return value: None.
 */
typedef void(*smap_iter_func)(const char *key, void *value, void *obj);

/*
 * Creates a SMap.
 *
 * Parameters:
 *
 * capacity: The number of top-level slots this SMap
 * should allocate. This parameter must be > 0.
 *
 * Return value: A pointer to a SMap object, 
 * or null if a new SMap could not be allocated.
 */
SMap *smap_ctor(unsigned int capacity);

/*
 * Releases all memory held by a SMap object.
 *
 * Parameters:
 *
 * smap: A pointer to a *SMap. This parameter cannot be null.
 * The pointer is set to NULL before returning.
 *
 * Return value: None.
 */
void smap_dtor(SMap **smap);

/*
 * Returns the value associated with the supplied key.
 *
 * Parameters:
 *
 * smap: A pointer to a SMap. This parameter cannot be null.
 *
 * key: A pointer to a key bytestring. This parameter cannot
 * be null.
 *
 * Return value: Either the value associated with key in smap
 * or else NULL (if the key is not in the map).
 *
 */
void *smap_get(const SMap *smap, const char *key);

/*
 * Associates a value with the supplied key. If the key is already
 * associated with a value, the previous value is returned.
 *
 * Parameters:
 *
 * smap: A pointer to a SMap. This parameter cannot be null.
 *
 * key: A pointer to a bytestring. This parameter
 * cannot be null. The key will be copied.
 *
 * value: A pointer to a value bytestring.  The pointer
 * will be saved, not copied.
 *
 * Return value: The previous value if key was already associated.
 *               NULL if the association is new.
 *               value on error.
 *
 */
void *smap_put(SMap *smap, const char *key, void *value);

/*
 * Returns the number of associations between keys and values.
 *
 * Parameters:
 *
 * smap: A pointer to a SMap. This parameter cannot be null.
 *
 * Return value: The number of associations between keys and values.
 */
int smap_get_count(const SMap *smap);

/*
 * An enumerator over all associations between keys and values.
 *
 * Parameters:
 *
 * smap: A pointer to a SMap. This parameter cannot be null.
 *
 * iter_func: A pointer to a callback function that will be
 * called by this procedure once for every key associated
 * with a value. This parameter cannot be null.
 *
 * obj: A pointer to a client-specific object. This parameter will be
 * passed back to the client's callback function. This parameter can
 * be null.
 *
 * Return value: -1 if an error occured.
 *              count of key:value pairs in the SMap otherwise.
 */
int smap_iter(const SMap *smap, smap_iter_func iter_func, void *obj);

#ifdef __cplusplus
}
#endif

#endif

